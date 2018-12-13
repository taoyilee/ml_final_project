from models.ae.ae_cnn import encoder, decoder, svhn_ae

# model = encoder()
# model.summary()
# model = decoder()
# model.summary()

model = svhn_ae()[0]
model.summary()
