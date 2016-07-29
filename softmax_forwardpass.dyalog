 z←t softmax_forwardpass x

 ⍝ Softmax layer - forward pass
 sm_y←sm_W+.×x
 tmp←sm_y[⍋sm_y]
 sm_ymax←tmp[⍴sm_y]
 sm_y←*(sm_y-sm_ymax)
 sm_y←sm_y÷(+/sm_y)
 sm_pred[;t]←sm_y
 sm_xt[;t]←we_o

 z←sm_y
