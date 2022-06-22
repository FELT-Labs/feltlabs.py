"""Module for managing encryption/decryption using keys."""
from base64 import a85decode, a85encode

from nacl.public import Box, PrivateKey, PublicKey


def export_public_key(private_key: bytes) -> bytes:
    """Export public key for contract join request.

    Args:
        private_key: bytes representing private key

    Returns:
        32 bytes representing public key
    """
    return bytes(PrivateKey(private_key).public_key)


def encrypt_nacl(public_key: bytes, data: bytes) -> bytes:
    """Encryption function using NaCl box compatible with MetaMask
    For implementation used in MetaMask look into: https://github.com/MetaMask/eth-sig-util

    Args:
        public_key: public key of recipient
        data: message data

    Returns:
        encrypted data
    """
    emph_key = PrivateKey.generate()
    enc_box = Box(emph_key, PublicKey(public_key))
    # Encryption is required to work with MetaMask decryption (requires utf8)
    data = a85encode(data)
    ciphertext = enc_box.encrypt(data)
    return bytes(emph_key.public_key) + ciphertext


def decrypt_nacl(private_key: bytes, data: bytes) -> bytes:
    """Decryption function using NaCl box compatible with MetaMask
    For implementation used in MetaMask look into: https://github.com/MetaMask/eth-sig-util

    Args:
        private_key: private key to decrypt with
        data: encrypted message data

    Returns:
        decrypted data
    """
    emph_key, ciphertext = data[:32], data[32:]
    box = Box(PrivateKey(private_key), PublicKey(emph_key))
    return a85decode(box.decrypt(ciphertext))
