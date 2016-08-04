 z←t softmax_backwardpass target

 :If 0=target
     target←1
 :EndIf
 :If t=1
     sm_target←target
 :Else
     sm_target←sm_target,target
 :EndIf
 :If t>1
     sm_d←sm_pred[;t-1]
 :Else
     sm_d←sm_pred[;t]
 :EndIf
 sm_d[target]←sm_d[target]-1
 :If t>1
     sm_dW←sm_d∘.×sm_xt[;t-1] ⍝ outer product
 :Else
     sm_dW←sm_d∘.×sm_xt[;t]
 :EndIf
 sm_delta←(⍉sm_W)+.×(n,1)⍴sm_d
 z←sm_delta
