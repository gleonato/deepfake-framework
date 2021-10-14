
Ideia 1

Gerar frames de um deepfake (img inteira)

Gerar frames do mesmo deepfaje atacado

Comparar frames no Forensically qto a atributos, ruidos, etc

Ideia 2

Treinar CNN 

A. Generating training data (Datasets) with non attacked and attacked deepfakes 

Compressão/Qualidade:
c0/raw - RAW
c23/hq
c40/lq



B. Types of Attack:
===================

White-box:

1. robust_fgsm(input_img, model, model_type, cuda = True, 
    max_iter = 100, alpha = 1/255.0, 
    eps = 16/255.0, desired_acc = 0.95,
    transform_set = {"gauss_noise", "gauss_blur", "translation", "resize"}
    ):


 
2. iterative_fgsm(input_img, model, model_type, cuda = True, max_iter = 100, alpha = 1/255.0, eps = 16/255.0, desired_acc = 0.99):
    input_var = autograd.Variable(input_img, requires_grad=True)

3. carlini_wagner_attack(input_img, model, model_type, cuda = True, 
    max_attack_iter = 500, alpha = 0.005, 
    const = 1e-3, max_bs_iter = 5, confidence = 20.0)

Blackbox:

1. black_box_attack(input_img, model, model_type, 
    cuda = True, max_iter = 100, alpha = 1/255.0, 
    eps = 16/255.0, desired_acc = 0.90, 
    transform_set = {"gauss_blur", "translation"})

1st test setup

qtde: 10 videos
compressao: raw
attack method: iterative_fgsm

Footnotes:

Tensorboard activation - https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#learn-more