 z←t emb_forwardpass x

 ⍝ Word-embedding - forward pass
 ⍝ return weights corresponding to the value sent as embedding
 :If 0=x
     x←1
 :EndIf
 :If 1=(∧/we_x='null')
     we_x←x
 :Else
     we_x←we_x,x
 :EndIf
 we_x[LAYERNUM;;t]←x
 we_o←we_W[LAYERNUM;;x]
 z←we_o
