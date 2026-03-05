from OpenSSL import crypto
import os

# 生成私钥
key = crypto.PKey()
key.generate_key(crypto.TYPE_RSA, 4096)

# 生成证书
cert = crypto.X509()
cert.get_subject().CN = '192.168.3.11'
cert.set_serial_number(1000)
cert.gmtime_adj_notBefore(0)
cert.gmtime_adj_notAfter(365*24*60*60)
cert.set_issuer(cert.get_subject())
cert.set_pubkey(key)
cert.sign(key, 'sha256')

# 保存证书
with open('cert.pem', 'wb') as f:
    f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))

# 保存私钥
with open('key.pem', 'wb') as f:
    f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

print('证书已生成：cert.pem, key.pem')