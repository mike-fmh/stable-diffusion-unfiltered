import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
import pandas as pd

import unicodedata
import re


logging.set_verbosity_error()


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def bi_bubble_sort(array, array_to_follow):
    n = len(array)
    for i in range(n):
        # Create a flag that will allow the function to
        # terminate early if there's nothing left to sort
        already_sorted = True
        # Start looking at each item of the list one by one,
        # comparing it with its adjacent value. With each
        # iteration, the portion of the array that you look at
        # shrinks because the remaining items have already been
        # sorted.
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                # If the item you're looking at is greater than its
                # adjacent value, then swap them
                array[j], array[j + 1] = array[j + 1], array[j]
                array_to_follow[j], array_to_follow[j+1] = array_to_follow[j+1], array_to_follow[j]
                # Since you had to swap two elements,
                # set the `already_sorted` flag to `False` so the
                # algorithm doesn't finish prematurely
                already_sorted = False
        # If there were no swaps during the last iteration,
        # the array is already sorted, and you can terminate
        if already_sorted:
            break


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def load_img(path, h0, w0):

    image = Image.open(path).convert("RGB")
    w, h = image.size

    #print(f"loaded input image of size ({w}, {h}) from {path}")
    if h0 is not None and w0 is not None:
        h, w = h0, w0

    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32

    #print(f"New image size ({w}, {h})")
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


config = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt", type=str, nargs="?", default="", help="the prompt to render"
)
parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/img2img-samples")
parser.add_argument("--init-img", type=str, nargs="?", help="path to the input image")

parser.add_argument(
    "--inpdir",
    type=str,
    default=None
)

parser.add_argument(
    "--skip_grid",
    action="store_true",
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action="store_true",
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=None,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=None,
    help="image width, in pixel space",
)
parser.add_argument(
    "--strength",
    type=float,
    default=0.75,
    help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--iterateseed",
    type=int,
    default=0,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="CPU or GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
)
parser.add_argument(
    "--format",
    type=str,
    help="output image format",
    choices=["jpg", "png"],
    default="png",
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim"],
    default="ddim",
)
opt = parser.parse_args()

tic = time.time()
os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir
grid_count = len(os.listdir(outpath)) - 1

if opt.seed == None:
    opt.seed = randint(0, 1000000)
seed_everything(opt.seed)

# Logging
logger(vars(opt), log_csv = "logs/img2img_logs.csv")

sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.cdevice = opt.device
model.unet_bs = opt.unet_bs
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = opt.device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    assert opt.prompt is not None
    prompt = opt.prompt
    data = [batch_size * [prompt]]
else:
    print(f"reading prompts from {opt.from_file}")
    prompt = opt.from_file.split("\\")[-1]
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = batch_size * list(data)
        data = list(chunk(sorted(data), batch_size))

# inpdir should contain frames of a video titled 0.png, 1.png...etc
files, nums = [], []
if opt.inpdir is None:
    files.append(opt.init_img.split("\\")[-1])
    nums.append(int(files[-1].split(".")[0]))
else:
    for file in tqdm(os.listdir(opt.inpdir), desc="Appending files"):
        filename = os.fsdecode(file)
        files.append(filename)
        # nums will mirror files, and will be used to sort both lists
        nums.append(int(files[-1].split(".")[0]))
        # print("\n" + filename)
    bi_bubble_sort(nums, files)
    print(len(files), "total files in inpdir")

assert 0.0 <= opt.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
t_enc = int(opt.strength * opt.ddim_steps)
print(f"target t_enc is {t_enc} steps")


if opt.precision == "autocast" and opt.device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

sample_path = outpath
if opt.inpdir is None:
    if opt.outdir == "outputs/img2img-samples":
        sample_path += f"/samples/result_{prompt}_--strength_{opt.strength}_--scale_{opt.scale}"
else:
    if opt.outdir == "outputs/img2img-samples":
        sample_path += "/samples/" + opt.inpdir.split("\\")[-1] + f"_{prompt}_--strength_{opt.strength}_--scale_{opt.scale}"

os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
with torch.no_grad():
    all_samples = list()
    for n in trange(opt.n_iter, desc="Sampling"):
        for file in tqdm(files, desc="files"):
            modelFS.to(opt.device)
            filename = file.split(".")[0]
            if os.path.isfile(f"{sample_path}/{filename}-1-{opt.seed}.png"):
                print(f"\nskipping {filename}-1-{opt.seed}.png already exists")
                continue
           # if ".png" not in file:
           #     continue
            assert os.path.isfile(os.path.join(opt.inpdir, file))
            init_image = load_img(os.path.join(opt.inpdir, file), opt.H, opt.W).to(opt.device)
            if opt.device != "cpu" and opt.precision == "autocast":
                model.half()
                modelCS.half()
                modelFS.half()
                init_image = init_image.half()
            init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
            init_latent = modelFS.get_first_stage_encoding(
                modelFS.encode_first_stage(init_image))  # move to latent space

            if opt.device != "cpu":
                mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                modelFS.to("cpu")
                while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                    time.sleep(1)
            for prompts in tqdm(data, desc="data", disable=True):
                with precision_scope("cuda"):
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                            time.sleep(1)

                    # encode (scaled latent)
                    z_enc = model.stochastic_encode(
                        init_latent,
                        torch.tensor([t_enc] * batch_size).to(opt.device),
                        opt.seed,
                        opt.ddim_eta,
                        opt.ddim_steps,
                    )
                    # decode it
                    samples_ddim = model.sample(
                        t_enc,
                        c,
                        z_enc,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        sampler=opt.sampler,
                        desc=file
                    )

                    modelFS.to(opt.device)
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        #img.save(os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}"))
                        fileexists = True
                        fname = filename
                        i = 0
                        while fileexists:
                            i += 1
                            use_fname = fname + f"-{i}-{opt.seed}"
                            use_fname = slugify(use_fname)
                            fileexists = os.path.isfile(f"{sample_path}/{use_fname}.png")
                        try:
                            img.save(f"{sample_path}/{use_fname}.png")
                        except:
                            img.save(f"{sample_path}/out.png")
                        base_count += 1
                        if opt.iterateseed == 1:
                            opt.seed += 1

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated(device=opt.device) / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated(device=opt.device) / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim

toc = time.time()

time_taken = (toc - tic) / 60.0

print(
    (
        "Samples finished in {0:.2f} minutes and exported to "
        + sample_path
    ).format(time_taken)
)
