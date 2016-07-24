 z←i output_forward x;h

 ⍝ output layer, forward pass
 h←i emb_forwardpass x
 h←i lstm_forwardpass h
 z←i softmax_forwardpass h
