 z←t output_backward delta

 h←t softmax_backwardpass delta
 h←t emb_backwardpass h
 z←t lstm_backwardpass h
