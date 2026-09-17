from pathlib import Path
from tb_pipeline import MODELS, RESULTS, gradcam
import tensorflow as tf
model=tf.keras.models.load_model(MODELS/'best_densenet121.keras')
paths=sorted((Path(__file__).resolve().parents[1]/'dataset/test').glob('*/*'))[:6]
out=RESULTS/'gradcam'; out.mkdir(parents=True,exist_ok=True)
for p in paths:
    target=out/f'{p.stem}_gradcam.png'; gradcam(model,p,target); print(target)
