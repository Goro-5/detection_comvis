import model_functions as mf

if __name__ == "__main__":
    # 初期モデルの学習と保存
    directory = "T:/Goro/ComputerVision/code/OpenCV/make_render_edge/joints_cropped_edge"
    model = mf.train_initial_model(directory, noise_factor=0.02)
    
    # 推論の例
    model_path = "initial_model.h5"
    img_path = "T:/Goro/ComputerVision/code/OpenCV/make_render_edge/joints_cropped_edge/PS-18SU/FRONT/x0_y-11_r120.0_cropped3678.png"
    prediction = mf.load_and_predict(model_path, img_path)
    print(f"Predicted class: {prediction}")
    
    # モデルの更新
    updated_model = mf.update_model(model_path, directory, noise_factor=0.02)
    
    # 新しいクラスの追加
    new_classes_directory = "path_to_new_classes_directory"
    expanded_model = mf.expand_model_for_new_classes(model_path, new_classes_directory, noise_factor=0.02)
