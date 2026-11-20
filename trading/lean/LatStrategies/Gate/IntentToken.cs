/*
 * LAT intent token: how a signed intent rides on a LEAN order.
 *
 *   order.Tag = "LATv1." + base64url(UTF-8 signing string) + "." + hex(r||s)
 *
 * base64url keeps the '|' delimited signing string unambiguous inside the tag; the signature is the
 * 128-hex-char IEEE-P1363 r||s encoding produced by packages/core/signing.py::IntentSigner.
 * The token is not secret (it is a public, verifiable authorisation), so it may appear in LEAN logs.
 */

using System;
using System.Text;

namespace Lat.Lean.Gate
{
    public static class IntentToken
    {
        public const string TagPrefix = "LATv1.";
        public const int MaxTagLength = 8192;

        public static string Encode(string signingString, string signatureHex)
        {
            if (signingString == null) throw new ArgumentNullException(nameof(signingString));
            if (!IsSignatureHex(signatureHex)) throw new ArgumentException("signature must be 128 lowercase hex chars", nameof(signatureHex));
            return TagPrefix + ToBase64Url(Encoding.UTF8.GetBytes(signingString)) + "." + signatureHex;
        }

        public static bool TryDecode(string tag, out string signingString, out string signatureHex, out string error)
        {
            signingString = null;
            signatureHex = null;
            error = null;
            if (string.IsNullOrEmpty(tag) || !tag.StartsWith(TagPrefix, StringComparison.Ordinal))
            {
                error = "unsigned order: no LAT intent token in the order tag";
                return false;
            }
            if (tag.Length > MaxTagLength)
            {
                error = "intent token too long";
                return false;
            }
            var body = tag.Substring(TagPrefix.Length);
            var dot = body.IndexOf('.');
            if (dot <= 0 || dot != body.LastIndexOf('.'))
            {
                error = "malformed intent token";
                return false;
            }
            var sig = body.Substring(dot + 1);
            if (!IsSignatureHex(sig))
            {
                error = "malformed intent signature (expected 128 lowercase hex chars)";
                return false;
            }
            byte[] raw;
            try
            {
                raw = FromBase64Url(body.Substring(0, dot));
            }
            catch (FormatException)
            {
                error = "malformed intent token payload";
                return false;
            }
            try
            {
                signingString = new UTF8Encoding(false, true).GetString(raw);
            }
            catch (ArgumentException)
            {
                error = "intent token payload is not valid UTF-8";
                return false;
            }
            signatureHex = sig;
            return true;
        }

        public static bool IsSignatureHex(string s)
        {
            if (s == null || s.Length != 128) return false;
            foreach (var c in s)
            {
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
            }
            return true;
        }

        public static string ToBase64Url(byte[] data) =>
            Convert.ToBase64String(data).TrimEnd('=').Replace('+', '-').Replace('/', '_');

        public static byte[] FromBase64Url(string s)
        {
            if (s.IndexOfAny(new[] { '+', '/', '=' }) >= 0) throw new FormatException("not base64url");
            var b = s.Replace('-', '+').Replace('_', '/');
            switch (b.Length % 4)
            {
                case 2: b += "=="; break;
                case 3: b += "="; break;
                case 1: throw new FormatException("bad base64url length");
            }
            return Convert.FromBase64String(b);
        }
    }
}
