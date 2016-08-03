 z←t softmax_forwardpass x;tmp

 ⍝ Softmax layer - forward pass
 sm_y←⍉sm_W+.×⍉x
 tmp←sm_y[⍋sm_y;]
 sm_ymax←tmp[;(¯1↑⍴sm_y)]
 sm_y←*(sm_y-sm_ymax)
 sm_y←sm_y÷(+/sm_y)
 :If 1=t
     sm_pred←sm_y
     sm_xt←x
 :Else
     sm_pred←sm_pred⍪sm_y
     sm_xt←sm_xt,x
 :EndIf

 z←sm_y
