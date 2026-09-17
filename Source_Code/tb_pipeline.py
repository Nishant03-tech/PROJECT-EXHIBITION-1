"""Complete TB-vs-Normal DenseNet121 pipeline. Run after installing requirements."""
from pathlib import Path
import os, json, random, shutil
import hashlib
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

SEED=42; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
IMG_SIZE=(224,224); BATCH=16
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'raw'; DATA=ROOT/'dataset'; RESULTS=ROOT/'results'; MODELS=ROOT/'models'

def collect():
    rows=[]
    seen=set()
    for cls,label in [('Normal',0),('Tuberculosis',1)]:
        for p in (RAW/cls).rglob('*'):
            if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}:
                digest=hashlib.sha256(p.read_bytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest); rows.append((p,label))
    return rows

def split_dataset():
    rows=collect(); paths=np.array([str(p) for p,l in rows]); labels=np.array([l for p,l in rows])
    tr, temp, ytr, ytemp=train_test_split(paths,labels,test_size=.30,stratify=labels,random_state=SEED)
    va, te, yva, yte=train_test_split(temp,ytemp,test_size=.50,stratify=ytemp,random_state=SEED)
    for name,arr,ys in [('train',tr,ytr),('validation',va,yva),('test',te,yte)]:
        for p,y in zip(arr,ys):
            dst=DATA/name/('TB' if y else 'Normal'); dst.mkdir(parents=True,exist_ok=True); shutil.copy2(p,dst/Path(p).name)
    return {"train":len(tr),"validation":len(va),"test":len(te)}

def datasets():
    aug=tf.keras.Sequential([tf.keras.layers.RandomRotation(.03),tf.keras.layers.RandomTranslation(.03,.03),tf.keras.layers.RandomZoom(.08)],name='medical_augmentation')
    def load(split, training=False):
        ds=tf.keras.utils.image_dataset_from_directory(DATA/split,image_size=IMG_SIZE,batch_size=BATCH,label_mode='binary',color_mode='rgb',shuffle=training,seed=SEED)
        norm=tf.keras.applications.densenet.preprocess_input
        def f(x,y):
            x=tf.cast(x,tf.float32); x=aug(x,training=True) if training else x; return norm(x),y
        return ds.map(f,num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return load('train',True),load('validation'),load('test')

def build_model():
    base=tf.keras.applications.DenseNet121(include_top=False,weights='imagenet',input_shape=(*IMG_SIZE,3)); base.trainable=False
    inp=tf.keras.Input(shape=(*IMG_SIZE,3)); x=base(inp,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); x=tf.keras.layers.Dropout(.35)(x); x=tf.keras.layers.Dense(128,activation='relu')(x); x=tf.keras.layers.Dropout(.25)(x); out=tf.keras.layers.Dense(1,activation='sigmoid',name='tb_probability')(x)
    model=tf.keras.Model(inp,out); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='binary_crossentropy',metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]); return model,base

def train():
    DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True); counts=split_dataset(); train_ds,val_ds,test_ds=datasets(); model,base=build_model(); ck=MODELS/'best_densenet121.keras'
    callbacks=[tf.keras.callbacks.ModelCheckpoint(ck,monitor='val_auc',mode='max',save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor='val_auc',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-7)]
    h1=model.fit(train_ds,validation_data=val_ds,epochs=20,callbacks=callbacks)
    for layer in base.layers[-40:]: layer.trainable=True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='binary_crossentropy',metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')])
    h2=model.fit(train_ds,validation_data=val_ds,epochs=10,callbacks=callbacks)
    history={k:[float(x) for x in h1.history.get(k,[])]+[float(x) for x in h2.history.get(k,[])] for k in set(h1.history)|set(h2.history)}; json.dump({'counts':counts,'history':history},open(RESULTS/'training_history.json','w'),indent=2); model.save(ck); return model,test_ds,history

def evaluate(model,test_ds,history):
    y=[]; p=[]
    for x,b in test_ds: y.extend(b.numpy().ravel()); p.extend(model.predict(x,verbose=0).ravel())
    y=np.array(y).astype(int); p=np.array(p); pred=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); spec=tn/(tn+fp) if tn+fp else 0
    metrics={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall_sensitivity':recall_score(y,pred,zero_division=0),'specificity':spec,'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p)}; json.dump(metrics,open(RESULTS/'metrics.json','w'),indent=2)
    open(RESULTS/'classification_report.txt','w').write(classification_report(y,pred,target_names=['Normal','TB'],zero_division=0))
    plt.figure(); plt.imshow([[tn,fp],[fn,tp]],cmap='Blues'); plt.xticks([0,1],['Normal','TB']); plt.yticks([0,1],['Normal','TB']); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); [plt.text(j,i,str(v),ha='center',va='center') for i,row in enumerate([[tn,fp],[fn,tp]]) for j,v in enumerate(row)]; plt.savefig(RESULTS/'confusion_matrix.png',dpi=160); plt.close()
    fpr,tpr,_=roc_curve(y,p); plt.figure(); plt.plot(fpr,tpr,label=f'AUC={metrics["roc_auc"]:.3f}'); plt.plot([0,1],[0,1],'--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.legend(); plt.savefig(RESULTS/'roc_curve.png',dpi=160); plt.close()
    for key,title,out in [('accuracy','Accuracy','accuracy_curve.png'),('loss','Loss','loss_curve.png')]:
        plt.figure(); plt.plot(history.get(key,[]),label='train'); plt.plot(history.get('val_'+key,[]),label='validation'); plt.xlabel('Epoch'); plt.ylabel(title); plt.legend(); plt.savefig(RESULTS/out,dpi=160); plt.close()
    return metrics

def gradcam(model,image_path,out_path):
    img=tf.keras.utils.load_img(image_path,target_size=IMG_SIZE,color_mode='rgb'); arr=tf.keras.utils.img_to_array(img); x=tf.keras.applications.densenet.preprocess_input(arr[None]); backbone=model.get_layer('densenet121'); pool=model.get_layer('global_average_pooling2d'); dense=model.get_layer('dense'); out_layer=model.get_layer('tb_probability')
    with tf.GradientTape() as tape:
        convout=backbone(x,training=False); z=pool(convout); z=model.get_layer('dropout')(z,training=False); z=dense(z); z=model.get_layer('dropout_1')(z,training=False); score=out_layer(z); loss=score[:,0]
    grads=tape.gradient(loss,convout); weights=tf.reduce_mean(grads,axis=(1,2)); cam=tf.maximum(tf.reduce_sum(convout*weights[:,None,None,:],axis=-1)[0],0); cam/=tf.reduce_max(cam)+1e-8; cam=tf.image.resize(cam[...,None],IMG_SIZE).numpy()[...,0]; heat=plt.cm.jet(cam)[...,:3]; overlay=.55*arr/255+.45*heat; fig,ax=plt.subplots(1,3,figsize=(12,4)); ax[0].imshow(arr.astype('uint8'),cmap='gray'); ax[1].imshow(cam,cmap='jet'); ax[2].imshow(np.clip(overlay,0,1)); [a.axis('off') for a in ax]; fig.suptitle(f'Prediction: {"TB" if score[0,0]>=.5 else "Normal"} | TB score: {float(score[0,0]):.4f}'); fig.savefig(out_path,dpi=160,bbox_inches='tight'); plt.close(fig)

def predict_xray(image_path):
    model=tf.keras.models.load_model(MODELS/'best_densenet121.keras'); img=tf.keras.utils.load_img(image_path,target_size=IMG_SIZE,color_mode='rgb'); x=tf.keras.applications.densenet.preprocess_input(tf.keras.utils.img_to_array(img)[None]); score=float(model.predict(x,verbose=0)[0,0]); out=RESULTS/'gradcam'/f'{Path(image_path).stem}_gradcam.png'; out.parent.mkdir(exist_ok=True); gradcam(model,image_path,out); return {'predicted_class':'TB' if score>=.5 else 'Normal','tb_score':score,'gradcam':str(out)}

if __name__=='__main__':
    model,test_ds,history=train(); print(json.dumps(evaluate(model,test_ds,history),indent=2))
