 z←i input_forward x;o
 ⍝ forward prop input layers (LSTM+Emb)

 o←i lstm_forwardpass x
 z←emb_forwardpass o
