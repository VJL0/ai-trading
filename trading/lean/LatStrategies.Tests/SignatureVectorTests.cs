using System.Linq;
using Lat.Lean.Gate;
using NUnit.Framework;

namespace Lat.Lean.Tests
{
    /// <summary>Cross-language: vectors signed by the Python IntentSigner must verify (or fail) identically in C#.</summary>
    [TestFixture]
    public class SignatureVectorTests
    {
        [Test]
        public void KeyIdMatchesPythonDerivation()
        {
            Assert.That(IntentSignatureVerifier.FromPem(Vectors.PublicKeyPem).KeyId, Is.EqualTo(Vectors.KeyId));
            Assert.That(IntentSignatureVerifier.FromPem(Vectors.OtherPublicKeyPem).KeyId, Is.EqualTo(Vectors.OtherKeyId));
        }

        [Test]
        public void VectorFileCoversValidAndInvalidSignatures()
        {
            Assert.That(Vectors.Cases.Count(c => c.ExpectSignatureValid), Is.GreaterThanOrEqualTo(3));
            Assert.That(Vectors.Cases.Count(c => !c.ExpectSignatureValid), Is.GreaterThanOrEqualTo(3));
        }

        [TestCaseSource(typeof(SignatureVectorTests), nameof(CaseNames))]
        public void SignatureVerdictMatchesPython(string name)
        {
            var c = Vectors.Case(name);
            using var verifier = IntentSignatureVerifier.FromPem(Vectors.PublicKeyPem);
            Assert.That(verifier.Verify(c.SigningString, c.Signature), Is.EqualTo(c.ExpectSignatureValid), name);
        }

        [Test]
        public void ForeignKeyVectorVerifiesOnlyUnderItsOwnKey()
        {
            var c = Vectors.Case("signed_by_other_key");
            using var other = IntentSignatureVerifier.FromPem(Vectors.OtherPublicKeyPem);
            using var ours = IntentSignatureVerifier.FromPem(Vectors.PublicKeyPem);
            Assert.That(other.Verify(c.SigningString, c.Signature), Is.True);
            Assert.That(ours.Verify(c.SigningString, c.Signature), Is.False);
        }

        [Test]
        public void ParsedFieldsComeFromTheSignedString()
        {
            var c = Vectors.Case("valid_buy");
            Assert.That(SignedIntent.TryParse(c.SigningString, out var si, out var err), Is.True, err);
            Assert.That(si.IntentId, Is.EqualTo(c.IntentId));
            Assert.That(si.ClientOrderId, Is.EqualTo(c.ClientOrderId));
            Assert.That(si.Quantity, Is.EqualTo(10m));
            Assert.That(si.LimitPrice, Is.EqualTo(450.10m));
            Assert.That(si.Side, Is.EqualTo("BUY"));
            Assert.That(si.KeyId, Is.EqualTo(Vectors.KeyId));
            Assert.That(si.Mode, Is.EqualTo("PAPER"));
        }

        [Test]
        public void MalformedSignaturesNeverVerify()
        {
            var c = Vectors.Case("valid_buy");
            using var v = IntentSignatureVerifier.FromPem(Vectors.PublicKeyPem);
            Assert.That(v.Verify(c.SigningString, c.Signature.ToUpperInvariant()), Is.False, "uppercase hex rejected");
            Assert.That(v.Verify(c.SigningString, c.Signature.Substring(2)), Is.False, "short");
            Assert.That(v.Verify(c.SigningString, ""), Is.False);
            Assert.That(v.Verify(c.SigningString, null), Is.False);
            Assert.That(v.Verify(c.SigningString + " ", c.Signature), Is.False, "trailing byte");
        }

        [Test]
        public void TokenRoundTripAndMalformedTokens()
        {
            var c = Vectors.Case("valid_buy");
            var tag = IntentToken.Encode(c.SigningString, c.Signature);
            Assert.That(IntentToken.TryDecode(tag, out var s, out var sig, out _), Is.True);
            Assert.That(s, Is.EqualTo(c.SigningString));
            Assert.That(sig, Is.EqualTo(c.Signature));
            Assert.That(IntentToken.TryDecode("", out _, out _, out _), Is.False);
            Assert.That(IntentToken.TryDecode("baseline: buy", out _, out _, out _), Is.False);
            Assert.That(IntentToken.TryDecode(tag + ".x", out _, out _, out _), Is.False);
            Assert.That(IntentToken.TryDecode("LATv1.!!!." + c.Signature, out _, out _, out _), Is.False);
        }

        public static string[] CaseNames() => Vectors.Cases.Select(c => c.Name).ToArray();
    }
}
