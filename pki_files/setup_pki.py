from Cryptodome.PublicKey import ECC

# generate server key
key = ECC.generate(curve='P-256')
f = open('server_key.pem', 'wt')
f.write(key.export_key(format='PEM'))
f.close()

# system-wide pk
key = ECC.generate(curve='P-256')
f = open('system_pk.pem', 'wt')
f.write(key.export_key(format='PEM'))
f.close()

# generate client keys
for i in range (512):
	key = ECC.generate(curve='P-256')
	hdr = 'client'+str(i)+'.pem'
	f = open(hdr, 'wt')
	f.write(key.export_key(format='PEM'))
	f.close()

