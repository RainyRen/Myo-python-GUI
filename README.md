# Myo

## myo_python 内部資料夾介紹
./config
    -> ini_deafult.yml 不要修改此文件，這是參數初始化設定，如果要修改初始化參數可以修改。在程式裏按reset按鈕會用到此文件
    -> ini_config.yml GUI在每次啓動的時候會讀取的配置文件，在GUI裏點set后會自動保存新的配置
    -> train_keras.yml 訓練keras模型時候用到的配置文件，主要是訓練的可選參數，模型的參數後來都寫死在模型程式裏
    -> train_trad.yml 訓練傳統模型時用到的配置文件，同樣主要為訓練的可選參數

./data
    用來使用GUI點save后存的資料，可存取的資料可以在設定頁選擇。應該根據選擇的sampling rate放到對應的資料夾下

。/exp
    用來存訓練模型的資料夾，每個模型都在對應的資料夾裏，以下列出重要的幾個模型

    -> JLW_stft_f4_dueling 使用JL、LK、WH三人5個session資料訓練好的模型，stft代表使用了frequency的feature(160維)，f4代表預測
    4個time steps，dueling代表使用了dueling的架構
        -> ./ey 存放EY的fine-tune模型，其餘姓名文件夾類推
        -> config.yml 為訓練該模型所選擇的配置參數
        -> logs.csv 為訓練模型每個epoch的loss記錄
        -> rnn_best.h5 為完整模型檔案，可以直接load
        -> rnn_weight.h5 為模型所有參數的weight，不可直接load，不包含元數據

        其餘圖片為用對應受試者的資料test后，任意選擇一段時間plot存儲

    -> JLW_stft_f4_L2 使用三人資料訓練，沒有deuling架構，L2代表最後FC layer使用linear activation與L相比就是參數略有不同，效果
    比L略好一點，可以忽略掉JLW_stft_f4_L

    -> J_stft_f4_dueling 為使用JL一個人的5session資料訓練的模型

    -> JLWEYYZ_stft_f4_dueling 為使用 JL、LK、WH、EY、YH、YC、ZE七人資料訓練的模型

    -> JLW_trad_f4 為使用傳統模型訓練的結果，裏面包含了linear model, SVR model, KNR model
        -> 每個人名的文件夾同樣為對應的fine-tune模型
        -> svr_model.p 為SVR的模型，使用pickle存儲，可用sklearn的load讀取

    所有資料夾后面加上_R的為右手的模型，沒有的話默認為左手資料訓練的模型
    ！！！慾强行使用左手的模型給右手用，比較好的做法是將myo先反向戴，在GUI的最後一頁裏手動選到right(日常再用的時候發現自動不准也
    可以手動設定對應手的位置)，反之使用右手模型在左手上采用同樣的做法！！！

。/images 存放了GUI界面所用到的圖片，有需要修改GUI内部小圖片時可以修改

。/models 設定模型架構的程式
    -> k_models 為kears的模型，裏面包含了所有的架構
    本來裏面還有tf的模型，後來論文裏也沒有寫，就拿掉了，跑混亂

。/myo myo底層的API，不建議更動，需要查看function的話可以去myo官網查看API的介紹(裏面原本的底層程式有錯，已經檢查並修復過)

。/utils 一些小的應用或者模塊可以放這邊
    -> data_io.py 在訓練模型的時候，導入訓練資料時會用到
    ！！！ row 41比較重要, *_L_*.csv意思為導入全部的資料，若要引入一個人的，可以改爲 LK_L*.csv！！！

### 外層單獨重要的程式講解
-> inference.py 訓練好模型后做test時候使用，如果要針對一個人做測試，記得先去data_io設定為一個人的全部資料
如何使用：
    python inference.py ==> MODEL_DIR_NAME 這邊要先設定好模型資料夾的名字，會test kears的模型
    python inference.py --mode test_trad ==> test 傳統的模型，一樣記得修改 MODEL_DIR_NAME

-> myo_feed.py 在GUI中選擇feed模型會用到的程式，feed模式只會讀取當前sample到的數值，不建議使用

-> myo_listen.py 在GUI中選擇listen會用到程式，也是目前主要維護的程式，配合GUI的程式相互溝通，此模式為監聽模式，有多個綫程，
IMU與EMG分別可以到達各自的sampling rate (50Hz, 200Hz)

-> myo_linstener.py，listen 模式下的底層程式，可以分別管理每一個綫程的内部，同時包含了計算手臂角度的程式

-> save2mat.py，可以將訓練資料轉換為matlab可以讀取的mat檔案，用於matlab程式

-> train.py 要訓練模型時會用到程式
如何使用：
    python train.py ==> 訓練keras模型，使用前先檢查train_keras.yml是否設定正確
    python train.py --mode train_trad ==> 訓練傳統模型，使用前先檢查train.trad.yml是否設定正確
    python train.py --fine_tune JLW_stft_f4_dueling ==> fine tune specfic model
    !!!在 fine tune 前應該先執行model_transfer.py 提取model的weight
