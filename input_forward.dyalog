 z←i input_forward x;o
 ⍝ forward prop input layers (LSTM+Emb)

 o←i emb_forwardpass x
 z←i lstm_forwardpass o
