 z←t emb_forwardpass x

 ⍝ Word-embedding - forward pass
 ⍝ return weight corresponding to the value sent as embedding
 we_x[LAYERNUM;;t]←x
 we_o←we_W[LAYERNUM;;x] 
 z←we_o
