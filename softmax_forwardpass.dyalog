 z←counter softmax_forwardpass x

 ⍝ Softmax layer - forward pass
 sm_y←sm_W+.×we_o[LAYERNUM;;]
 tmp←sm_y[⍋sm_y]
 sm_ymax←tmp[⍴sm_y]
 sm_y←*(sm_y-sm_ymax)
 sm_y←sm_y÷(+/sm_y)
 sm_pred[;counter]←sm_y
 sm_xt[;counter]←we_o

 z←sm_y
