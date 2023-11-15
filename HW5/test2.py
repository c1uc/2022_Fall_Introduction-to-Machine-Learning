from gen_captcha import _gen_captcha

GEN_PATH = "./gen"
num_char = 2
img_size = [(0, 0), (72, 72), (72, 72), (96, 72)]

_gen_captcha(f"{GEN_PATH}/", num_char, 5, width=img_size[1][0], height=img_size[1][1])
