import tensorflow as tf
m=tf.keras.models.load_model('/home/ubuntu/TB_Detection/models/best_densenet121.keras')
for i,l in enumerate(m.layers): print(i,l.name,type(l).__name__,getattr(l,'output_shape',None))
