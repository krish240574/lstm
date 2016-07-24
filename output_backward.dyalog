 z←i output_backward delta

 h←i softmax_backwardpass delta
 h←i emb_backwardpass h
 z←i lstm_backwardpass h
