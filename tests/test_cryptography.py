"""Test encryption/decription mechanism."""
from nacl.public import PrivateKey

from feltlabs.core.cryptography import decrypt_nacl, encrypt_nacl, export_public_key


def test_cryptography():
    priv_key = PrivateKey.generate()
    priv_key_b = bytes(priv_key)

    data = b"test data"

    pub_key = export_public_key(priv_key_b)
    assert pub_key == bytes(priv_key.public_key)

    ciphertext = encrypt_nacl(pub_key, data)
    plaintext = decrypt_nacl(priv_key_b, ciphertext)

    assert plaintext == b"test data"
