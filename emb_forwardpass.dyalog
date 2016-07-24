 z←counter emb_forwardpass x

 ⍝ Word-embedding - forward pass
 we_x[LAYERNUM;;counter]←ht[LAYERNUM;;counter] ⍝ output of LSTM layer
 we_o←we_W[we_x[LAYERNUM;;counter]] ⍝ output of embedding layer
 z←we_o
