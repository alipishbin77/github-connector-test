"""Print a fresh RS256 signing key for AETHER_JWT_PRIVATE_KEY_PEM.

    python -m app.keygen

Every clearinghouse replica must share the same key (sellers verify delivery
tokens against its public half at /.well-known/jwks.json).
"""

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

if __name__ == "__main__":
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    print(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode(),
        end="",
    )
