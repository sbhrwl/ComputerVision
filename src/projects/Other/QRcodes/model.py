!pip install PyQRCode
!pip install pypng
''' dataset path https://www.kaggle.com/coledie/qr-codes'''
p = []
for dir_name, _, filenames in os.walk('../input/qr-codes/qr_dataset/'):
    for file_name in filenames:
        p += [os.path.join(dir_name, file_name)]
''' Total lenght '''
print(len(p))
''' Create and Read QR cords '''

''' classes '''
t = ['0123456789','Happy_New_Year','Tokyo_2020_Olympics','Pandemic']
for i in t:
    c = pyqrcode.QRCode(item,error='M')
    c.png('./'+ i + '.png', scale=6)
''' plotting images '''

fig,axs = plt.subplots(2,2,figsize=(8,8))
for i in range(4):
    ''' reading images '''
    img = cv2.imread('./'+ t[i] + '.png')
    r = i // 2
    c = i % 2
    axs[r][c].set_xticks([])
    axs[r][c].set_yticks([])
    axs[r][c].set_title(t[i])
    ax=axs[r][c].imshow(img)
plt.show()
'''Reading method qrdec2_cv2'''
def qr_dec(img):
    qr_c = cv2.QRCodeDetector()
    r, dec_info, p, str_qr = qr_c.detectAndDecodeMulti(img)
    res = [r, dec_info, p, str_qr]
    if r== True:
        return res[1][0]
    else:
        return 'False'
for i in range(4):
    ''' reading images '''
    img = cv2.imread('./'+ t[i] +'.png')
    dc = qr_dec(img)
    print(dc)
''' Dataset readability ANA '''
ans = []
for i in p:
    ans += [i.split('/')[4][0:-7]]
print(ans[0:10])
dc = []
for i in range(10000):
    img = cv2.imread(p[i])
    dc += [qr_dec(img)]
print(DC[0:10])
''' accuracy score '''
accuracy_score(ans, dc)
