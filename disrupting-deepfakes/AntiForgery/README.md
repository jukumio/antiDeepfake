
## Preparation
**CelebA Dataset**

```
bash download.sh celeba
```
**StarGAN Model**

```
bash download.sh pretrained-celeba-256x256
```

## Attack Testing

Here is a simple example of  testing our method to attack StarGAN on the CelebA dataset.
```
# Test
python main.py --mode test --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_256/models' --result_dir='./results' --test_iters 200000 --attack_iters 100 --batch_size 1
```
