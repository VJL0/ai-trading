/*
 * ECDSA P-256 / SHA-256 verification of LAT order intents (public key only: LEAN can never mint intents).
 * Matches packages/core/signing.py: signature = hex(r||s) (IEEE P1363 fixed-field), message = UTF-8 bytes
 * of the signing string, key_id = "k_" + sha256_hex(SubjectPublicKeyInfo DER)[:16].
 */

using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace Lat.Lean.Gate
{
    public sealed class IntentSignatureVerifier : IDisposable
    {
        public const string SignatureAlg = "ECDSA_P256_SHA256";
        public const string PubKeyPathEnv = "LAT_INTENT_PUBKEY_PATH";
        public const string PubKeyPathConfig = "lat-intent-pubkey-path";
        private const string NistP256Oid = "1.2.840.10045.3.1.7";

        private readonly ECDsa _key;

        /// <summary>"k_" + first 16 hex chars of SHA-256 over the SubjectPublicKeyInfo DER.</summary>
        public string KeyId { get; }

        private IntentSignatureVerifier(ECDsa key, string keyId)
        {
            _key = key;
            KeyId = keyId;
        }

        public static IntentSignatureVerifier FromPem(string pem)
        {
            if (string.IsNullOrWhiteSpace(pem)) throw new ArgumentException("empty public key PEM");
            const string begin = "-----BEGIN PUBLIC KEY-----";
            const string end = "-----END PUBLIC KEY-----";
            var b = pem.IndexOf(begin, StringComparison.Ordinal);
            var e = pem.IndexOf(end, StringComparison.Ordinal);
            if (b < 0 || e < b) throw new CryptographicException("expected a 'BEGIN PUBLIC KEY' (SubjectPublicKeyInfo) PEM");
            var body = pem.Substring(b + begin.Length, e - b - begin.Length);
            var der = Convert.FromBase64String(body.Replace("\r", "").Replace("\n", "").Trim());

            var key = ECDsa.Create();
            key.ImportSubjectPublicKeyInfo(der, out var read);
            if (read != der.Length) throw new CryptographicException("trailing bytes after SubjectPublicKeyInfo");
            var curve = key.ExportParameters(false).Curve;
            if (key.KeySize != 256 || curve.Oid?.Value != NistP256Oid)
            {
                key.Dispose();
                throw new CryptographicException("intent verification key must be ECDSA P-256 (secp256r1)");
            }
            var keyId = "k_" + Convert.ToHexString(SHA256.HashData(der)).ToLowerInvariant().Substring(0, 16);
            return new IntentSignatureVerifier(key, keyId);
        }

        public static IntentSignatureVerifier FromPemFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("no intent public key path configured");
            if (!File.Exists(path)) throw new FileNotFoundException("intent public key not found", path);
            return FromPem(File.ReadAllText(path));
        }

        /// <summary>
        /// Resolves the public key path: environment LAT_INTENT_PUBKEY_PATH first, then LEAN config
        /// 'lat-intent-pubkey-path'. Throws (fail closed) when neither is set or the key is invalid.
        /// </summary>
        public static IntentSignatureVerifier FromEnvironmentOrConfig(Func<string, string> configGet)
        {
            var path = Environment.GetEnvironmentVariable(PubKeyPathEnv);
            if (string.IsNullOrWhiteSpace(path) && configGet != null)
            {
                path = configGet(PubKeyPathConfig);
            }
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new InvalidOperationException(
                    $"LAT gate: no intent public key configured ({PubKeyPathEnv} or config '{PubKeyPathConfig}'); refusing to start");
            }
            return FromPemFile(path);
        }

        public bool Verify(string signingString, string signatureHex)
        {
            if (signingString == null || !IntentToken.IsSignatureHex(signatureHex)) return false;
            byte[] sig;
            try
            {
                sig = Convert.FromHexString(signatureHex);
            }
            catch (FormatException)
            {
                return false;
            }
            if (sig.Length != 64) return false;
            try
            {
                return _key.VerifyData(Encoding.UTF8.GetBytes(signingString), sig, HashAlgorithmName.SHA256,
                    DSASignatureFormat.IeeeP1363FixedFieldConcatenation);
            }
            catch (CryptographicException)
            {
                return false;
            }
        }

        public void Dispose() => _key.Dispose();
    }
}
