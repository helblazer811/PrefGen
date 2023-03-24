from prefgen.methods.attribute_classifiers.facial_expression.train import test_model, load_facenet_model
import os

if __name__ == "__main__":
    PRETRAINED_PATH = os.path.join(os.environ["PREFGEN_ROOT"], 'prefgen/pretrained/')

    facenet_path = os.path.join(
        PRETRAINED_PATH, 
        'face_expression/facenet_modified_best.pt'
    )

    model = load_facenet_model(save_path=facenet_path).cuda()
    test_model(model)