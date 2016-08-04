 z←t output_forward x;h

 ⍝ output layer, forward pass
 h←t emb_forwardpass x
 h←t lstm_forwardpass h
 z←t softmax_forwardpass 
