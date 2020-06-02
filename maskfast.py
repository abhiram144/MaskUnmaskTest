from fastai import *
from fastai.vision import *
import pathlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename


path = pathlib.Path(__file__).parent / "Data/598475_1075843_bundle_archive/Mask_Datasets/"
maskImgpath = pathlib.Path(path) / 'Validation/Mask'
noMaskImgpath = pathlib.Path(path) / 'Validation/No_mask'

tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path,train="Train", valid="validation" ,ds_tfms=tfms,size=224,bs=16)



def trainlearner():
    
    data.show_batch(rows=3,figsize=(5,5))

    size = 224

    learn = create_cnn(data,models.resnet34,metrics=error_rate)

    learn.model_dir= pathlib.Path(path).parent.parent / 'Learning'

    learn.fit_one_cycle(4)

    learn.save('mask-final')

    learn.unfreeze()

    learn.lr_find()

    # learn.recorder.plot()

    # interp = ClassificationInterpretation.from_learner(learn)

    # interp.plot_confusion_matrix()


def testlearner():
    
    data = ImageDataBunch.from_folder(path, train="Train", valid="validation", ds_tfms=tfms,size=224)
    learn = cnn_learner(data,models.resnet34).load(pathlib.Path(path).parent.parent / 'Learning' / 'mask-final')
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()
    img = open_image(fn= filename)
    prediction,pred_idx,output = learn.predict(img)
    print(prediction,pred_idx,output)

if __name__ == "__main__":
    trainlearner()
    testlearner()

    
    

