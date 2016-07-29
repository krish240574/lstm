 z←t emb_forwardpass x

 ⍝ Word-embedding - forward pass
 ⍝ return weight corresponding to the value sent as embedding
 we_x[LAYERNUM;;t]←ht[LAYERNUM;;t] ⍝ output of LSTM layer
 we_o←we_W[we_x[LAYERNUM;;t]] ⍝ output of embedding layer
 z←we_o
