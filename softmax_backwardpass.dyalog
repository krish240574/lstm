 z←counter softmax_backwardpass delta

 :If counter>1
     sm_d←sm_pred[;counter-1]
 :Else
     sm_d←sm_pred[;counter]
 :EndIf
 sm_d[target]←sm_d[target]-1
 :If counter>1
     sm_dW←sm_d×.∘sm_xt[;counter-1] ⍝ outer product
 :Else
     sm_dW←sm_d×.∘sm_xt[;counter]
 :EndIf
 sm_delta←sm_W+.×sm_d
 z←sm_delta
