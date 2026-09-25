import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = Path(os.environ.get("WATERMARK_MODELS_ROOT", PROJECT_ROOT / "third_party"))
VIDEOSEAL_ROOT = Path(os.environ.get("VIDEOSEAL_ROOT", AGENT_ROOT / "videoseal"))
STEGASTAMP_ROOT = Path(os.environ.get("STEGASTAMP_ROOT", AGENT_ROOT / "StegaStamp-pytorch"))
def build_argparser():
    parser = argparse.ArgumentParser(
        description="Shared decoder/encoder model and checkpoint configuration.",
    )
    parser.add_argument(
        "--watermarked-image",
        default=os.environ.get("DEFAULT_WATERMARKED_IMAGE", ""),
        help="Path to the watermarked image to attack.",
    )
    parser.add_argument(
        "--ranking-csv",
        default=os.environ.get("DEFAULT_RANKING_CSV"),
        help="CSV produced by compute_second_similarity.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("DEFAULT_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "single")),
        help="Directory for generated evaluation artifacts and logs.",
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=None,
        help="Method name to remove from the training decoder candidate ranking. Can be repeated.",
    )
    parser.add_argument(
        "--force-method",
        action="append",
        default=None,
        help="Method name to always include as a training decoder candidate after ranking selection. Can be repeated.",
    )
    parser.add_argument(
        "--eval-method",
        action="append",
        default=None,
        help="Extra decoder method to evaluate on the final adversarial image only. Can be repeated.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=None, help="Optional square resize before attack.")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument("--mbrs-model-path", default=str(AGENT_ROOT / "MBRS/results/MBRS_256_m256/models/EC_42.pth"))
    parser.add_argument("--mbrs-h", type=int, default=256)
    parser.add_argument("--mbrs-w", type=int, default=256)
    parser.add_argument("--mbrs-message-length", type=int, default=256)
    parser.add_argument("--mbrs-with-diffusion", action="store_true")
    parser.add_argument("--mbrs-noise-layer", action="append", default=None)

    parser.add_argument(
        "--lightweightmark-model-path",
        default=str(AGENT_ROOT / "LightweightMark/experiments/DO/CombinedNoise_DO/models/200.pt"),
    )
    parser.add_argument(
        "--lightweightmark-mode", default="DO", choices=["DO", "PH", "MSE"]
    )
    parser.add_argument("--lightweightmark-message-length", type=int, default=64)
    parser.add_argument("--lightweightmark-h", type=int, default=128)
    parser.add_argument("--lightweightmark-w", type=int, default=128)

    parser.add_argument("--videoseal-model-name", default="videoseal")
    parser.add_argument(
        "--videoseal-message-length",
        type=int,
        default=None,
        help="VideoSeal bit length. Defaults to the loaded model capacity.",
    )
    parser.add_argument("--videoseal-short-edge", type=int, default=256)
    parser.add_argument("--videoseal-lowres-attenuation", action="store_true")
    parser.add_argument("--chunkyseal-model-name", default="chunkyseal")
    parser.add_argument(
        "--chunkyseal-message-length",
        type=int,
        default=None,
        help="ChunkySeal bit length. Defaults to the loaded model capacity.",
    )
    parser.add_argument("--chunkyseal-short-edge", type=int, default=512)
    parser.add_argument("--chunkyseal-lowres-attenuation", action="store_true")

    parser.add_argument(
        "--hidden-options-file",
        default=str(AGENT_ROOT / "HiDDeN/experiments/dropout-0.55-0.6/options-and-config.pickle"),
    )
    parser.add_argument(
        "--hidden-checkpoint-file",
        default=str(AGENT_ROOT / "HiDDeN/experiments/dropout-0.55-0.6/checkpoints/epoch-300.pyt"),
    )

    parser.add_argument("--pimog-image-size", type=int, default=128)
    parser.add_argument("--pimog-embedding-epoch", type=int, default=99)
    parser.add_argument("--pimog-distortion", default="ScreenShooting", choices=["Identity", "ScreenShooting"])
    parser.add_argument("--pimog-model-save-dir", default=str(AGENT_ROOT / "PIMoG/models"))
    parser.add_argument("--pimog-model-name", default="Encoder_Decoder_Model")

    parser.add_argument(
        "--stegastamp-model-path",
        default=str(STEGASTAMP_ROOT / "asset/best.pth"),
    )
    parser.add_argument("--stegastamp-image-size", type=int, default=400)
    parser.add_argument("--stegastamp-message-length", type=int, default=100)

    parser.add_argument("--cin-options-file", default=str(AGENT_ROOT / "CIN/codes/options/opt.yml"))
    parser.add_argument("--cin-checkpoint", default=str(AGENT_ROOT / "CIN/pth/cinNet&nsmNet.pth"))
    parser.add_argument(
        "--cin-decoder-branch",
        default="auto",
        choices=["auto", "0", "1"],
        help="CIN decoding branch: auto uses NSM, 0 forces invertible decoder, 1 forces NIAM/JPEG decoder.",
    )

    parser.add_argument(
        "--fin-noise-type",
        default="HEAVY",
        choices=["JPEG", "HEAVY"],
        help="FIN variant used by the backward-compatible 'fin' method.",
    )
    parser.add_argument("--fin-fed-checkpoint", default=None)
    parser.add_argument("--fin-inl-checkpoint", default=None)
    parser.add_argument("--fin-heavy-fed-checkpoint", default=None)
    parser.add_argument("--fin-heavy-inl-checkpoint", default=None)
    parser.add_argument("--fin-jpeg-fed-checkpoint", default=None)

    parser.add_argument("--trustmark-model-type", default="P", choices=["B", "C", "P", "Q"])
    parser.add_argument(
        "--trustmark-encoding-type",
        default="BCH_5",
        choices=["BCH_5", "BCH_4", "BCH_3", "BCH_SUPER"],
    )

    parser.add_argument("--invismark-ckpt", default=str(AGENT_ROOT / "InvisMark/paper.ckpt"))

    parser.add_argument("--rosteals-config", default=str(AGENT_ROOT / "RoSteALS/models/VQ4_mir_inference.yaml"))
    parser.add_argument(
        "--rosteals-weight",
        default=str(AGENT_ROOT / "RoSteALS/models/RoSteALS/epoch=000017-step=000449999.ckpt"),
    )
    parser.add_argument("--rosteals-image-size", type=int, default=256)
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_sys_path(path):
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def remove_sys_path(path):
    path = str(path)
    while path in sys.path:
        sys.path.remove(path)


def purge_modules(prefixes):
    for name in list(sys.modules.keys()):
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            sys.modules.pop(name, None)


def prepare_fin_import_path():
    purge_modules(["model", "models", "block", "utils", "noise_layers"])
    remove_sys_path(AGENT_ROOT / "HiDDeN")
    remove_sys_path(AGENT_ROOT / "InvisMark")
    remove_sys_path(AGENT_ROOT / "CIN/codes")
    remove_sys_path(AGENT_ROOT / "PIMoG")
    remove_sys_path(AGENT_ROOT / "MBRS")
    add_sys_path(AGENT_ROOT / "FIN")


def freeze_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_fin_checkpoint(path, network, device):
    """Load a FIN checkpoint without importing the ambiguous top-level utils package.

    FIN's ``utils`` directory is a namespace package, while LightweightMark exposes
    a regular ``utils`` package with the opposite ``load`` argument order. In a
    long-lived batch process, ``from utils.utils import load`` can therefore resolve
    to LightweightMark and pass the FED module itself to ``torch.load``.
    """
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("net", checkpoint)
    state_dict = {
        key: value for key, value in state_dict.items() if "tmp_var" not in key
    }
    network.load_state_dict(state_dict)


def checkpoint_target_key(key, current_state):
    if key in current_state:
        return key
    if key.startswith("module."):
        stripped = key.removeprefix("module.")
        if stripped in current_state:
            return stripped
    prefixed = f"module.{key}"
    if prefixed in current_state:
        return prefixed
    return None


def gradient_cosine(first, second, eps=1e-12):
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_norm = first_flat.norm()
    second_norm = second_flat.norm()
    if first_norm <= eps or second_norm <= eps:
        return 0.0
    cosine = F.cosine_similarity(first_flat, second_flat, dim=0)
    return float(cosine.detach().item())


def resize_if_needed(image, height, width):
    if image.shape[-2:] == (height, width):
        return image
    return F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)


class DecoderWrapper:
    def __init__(self, name, model, target_value=0.5, threshold=0.5):
        self.name = name
        self.model = model
        self.target_value = target_value
        self.threshold = threshold
        self.clean_decoded = None
        self.clean_bits = None

    def decode(self, image):
        raise NotImplementedError

    def bits(self, decoded):
        return decoded.detach().gt(self.threshold)

    def probability_values(self, decoded):
        if self.threshold == 0.0 and self.target_value == 0.0:
            probabilities = (decoded + 1.0) * 0.5
        else:
            probabilities = decoded
        return torch.nan_to_num(probabilities, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def clean_bit_confidence(self, decoded):
        return (decoded.detach() - self.threshold).abs()

    def cache_clean_bits(self, image):
        with torch.no_grad():
            decoded = self.decode(image)
            self.clean_decoded = decoded.detach()
            self.clean_bits = self.bits(decoded)

    def bit_error_rate(self, decoded):
        if self.clean_bits is None:
            raise RuntimeError(f"clean bits were not cached for decoder {self.name}")
        bits = self.bits(decoded)
        compare_len = min(bits.shape[1], self.clean_bits.shape[1])
        return float(bits[:, :compare_len].ne(self.clean_bits[:, :compare_len]).float().mean().item())


class MBRSDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["network", "utils"])
        remove_sys_path(AGENT_ROOT / "HiDDeN")
        remove_sys_path(AGENT_ROOT / "PIMoG")
        remove_sys_path(AGENT_ROOT / "InvisMark")
        remove_sys_path(AGENT_ROOT / "CIN/codes")
        remove_sys_path(AGENT_ROOT / "FIN")
        remove_sys_path(AGENT_ROOT / "trustmark/python")
        remove_sys_path(AGENT_ROOT / "RoSteALS")
        remove_sys_path(AGENT_ROOT / "RoSteALS/taming-transformers-master")
        add_sys_path(AGENT_ROOT / "MBRS")
        from network.Encoder_MP_Decoder import EncoderDecoder, EncoderDecoder_Diffusion

        noise_layers = args.mbrs_noise_layer.copy() if args.mbrs_noise_layer else ["Identity()"]
        if args.mbrs_with_diffusion:
            model = EncoderDecoder_Diffusion(args.mbrs_h, args.mbrs_w, args.mbrs_message_length, noise_layers)
        else:
            model = EncoderDecoder(args.mbrs_h, args.mbrs_w, args.mbrs_message_length, noise_layers)
        state_dict = torch.load(args.mbrs_model_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        freeze_eval(model.to(args.device))
        super().__init__("MBRS", model, target_value=0.5, threshold=0.5)
        self.height = args.mbrs_h
        self.width = args.mbrs_w

    def decode(self, image):
        return self.model.decoder(resize_if_needed(image, self.height, self.width))


class HiddenDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["model", "models", "noise_layers", "utils"])
        remove_sys_path(AGENT_ROOT / "PIMoG")
        remove_sys_path(AGENT_ROOT / "InvisMark")
        add_sys_path(AGENT_ROOT / "HiDDeN")
        from model.hidden import Hidden
        from noise_layers.noiser import Noiser
        import utils

        _, hidden_config, noise_config = utils.load_options(args.hidden_options_file)
        noiser = Noiser(noise_config, args.device)
        checkpoint = torch.load(args.hidden_checkpoint_file, map_location=args.device)
        model = Hidden(hidden_config, args.device, noiser, None)
        utils.model_from_checkpoint(model, checkpoint)
        freeze_eval(model.encoder_decoder.to(args.device))
        super().__init__("hidden", model.encoder_decoder, target_value=0.5, threshold=0.5)
        self.height = hidden_config.H
        self.width = hidden_config.W

    def decode(self, image):
        return self.model.decoder(resize_if_needed(image, self.height, self.width))


class PIMoGDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["model", "Noise_Layer"])
        add_sys_path(AGENT_ROOT / "PIMoG")
        from model import Encoder_Decoder

        model = Encoder_Decoder(args.pimog_distortion).to(args.device)
        checkpoint_path = (
            Path(args.pimog_model_save_dir)
            / args.pimog_distortion
            / f"{args.pimog_model_name}_mask_{args.pimog_embedding_epoch}.pth"
        )
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        freeze_eval(model)
        super().__init__("PIMoG", model, target_value=0.5, threshold=0.5)
        self.image_size = args.pimog_image_size

    def decode(self, image):
        image = resize_if_needed(image, self.image_size, self.image_size)
        # PIMoG's original scripts load images with cv2, so the trained decoder expects BGR channel order.
        image = image[:, [2, 1, 0], :, :]
        return self.model.Decoder(image.float())


def load_stegastamp_component(args, component):
    add_sys_path(STEGASTAMP_ROOT)
    from stegastamp.models import StegaStampDecoder as ModelDecoder
    from stegastamp.models import StegaStampEncoder as ModelEncoder

    model_class = ModelDecoder if component == "decoder" else ModelEncoder
    model = model_class(
        height=args.stegastamp_image_size,
        width=args.stegastamp_image_size,
        secret_size=args.stegastamp_message_length,
    ).to(args.device)
    checkpoint = torch.load(
        args.stegastamp_model_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or component not in checkpoint:
        raise KeyError(
            f"StegaStamp checkpoint {args.stegastamp_model_path} has no '{component}' state dict"
        )
    model.load_state_dict(checkpoint[component], strict=True)
    return freeze_eval(model)


class StegaStampDecoder(DecoderWrapper):
    def __init__(self, args):
        model = load_stegastamp_component(args, "decoder")
        super().__init__("stegastamp", model, target_value=0.0, threshold=0.0)
        self.image_size = args.stegastamp_image_size

    def decode(self, image):
        image = resize_if_needed(image, self.image_size, self.image_size)
        image_01 = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
        return self.model(image_01)

    def clean_bit_confidence(self, decoded):
        return (torch.sigmoid(decoded.detach()) - 0.5).abs()

    def probability_values(self, decoded):
        return torch.sigmoid(decoded).clamp(0.0, 1.0)

class CINDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["models", "utils"])
        add_sys_path(AGENT_ROOT / "CIN/codes")
        from models.CIN import CIN
        from utils.yml import dict_to_nonedict, parse_yml

        option_yml = parse_yml(args.cin_options_file)
        self.opt = dict_to_nonedict(option_yml)
        self.opt["train"]["batch_size"] = 1
        self.opt["train"]["num_workers"] = 0
        self.opt["path"]["folder_temp"] = str(Path(args.output_dir).resolve() / "cin_temp")

        model = CIN(self.opt, args.device).to(args.device)
        checkpoint = torch.load(args.cin_checkpoint, map_location=args.device)
        state_dict = checkpoint.get("cinNet", checkpoint)
        current_state = model.state_dict()
        loaded_state = {}
        for key, value in state_dict.items():
            target_key = checkpoint_target_key(key, current_state)
            if target_key is not None:
                loaded_state[target_key] = value
        if not loaded_state:
            raise RuntimeError(f"No CIN checkpoint keys matched model state from {args.cin_checkpoint}")
        current_state.update(loaded_state)
        model.load_state_dict(current_state, strict=False)
        freeze_eval(model)
        target_value = 0.5 if self.opt["datasets"]["msg"]["mod_a"] else 0.0
        threshold = 0.5 if self.opt["datasets"]["msg"]["mod_a"] else 0.0
        super().__init__("CIN", model, target_value=target_value, threshold=threshold)
        self.height = self.opt["datasets"]["H"]
        self.width = self.opt["datasets"]["W"]
        self.decoder_branch = "auto" if args.cin_decoder_branch == "auto" else int(args.cin_decoder_branch)

    def decode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        if self.decoder_branch == "auto":
            pre_noise = self.model.nsm(image)
        else:
            pre_noise = torch.tensor(float(self.decoder_branch), device=image.device)
        _, msg_1, msg_2, msg = self.model.test_decoder(image, pre_noise)
        return msg_1 if msg_1 is not None else (msg_2 if msg_2 is not None else msg)

    def bits(self, decoded):
        if self.opt["datasets"]["msg"]["mod_a"]:
            return decoded.detach().round().clamp(0, 1).gt(0.5)
        if self.opt["datasets"]["msg"]["mod_b"]:
            return decoded.detach().round().clamp(-1, 1).gt(0.0)
        raise ValueError("Unsupported CIN message mode.")


class FINDecoder(DecoderWrapper):
    def __init__(self, args, noise_type=None, method_name=None):
        prepare_fin_import_path()
        from models.encoder_decoder import FED, INL

        noise_type = (noise_type or args.fin_noise_type).upper()
        if noise_type not in {"HEAVY", "JPEG"}:
            raise ValueError(f"Unsupported FIN noise type: {noise_type}")
        explicit_variant = method_name is not None
        variant_fed_checkpoint = getattr(args, f"fin_{noise_type.lower()}_fed_checkpoint", None)
        generic_fed_checkpoint = None if explicit_variant else args.fin_fed_checkpoint
        fed_checkpoint_value = variant_fed_checkpoint or generic_fed_checkpoint
        fed_checkpoint = Path(fed_checkpoint_value) if fed_checkpoint_value else (
            AGENT_ROOT / "FIN/experiments" / noise_type / "FED.pt"
        )
        inl_checkpoint = None
        if noise_type == "HEAVY":
            variant_inl_checkpoint = getattr(args, "fin_heavy_inl_checkpoint", None)
            generic_inl_checkpoint = None if explicit_variant else args.fin_inl_checkpoint
            inl_checkpoint_value = variant_inl_checkpoint or generic_inl_checkpoint
            inl_checkpoint = Path(inl_checkpoint_value) if inl_checkpoint_value else (
                AGENT_ROOT / "FIN/experiments/HEAVY/INL.pt"
            )

        fed = FED().to(args.device)
        load_fin_checkpoint(fed_checkpoint, fed, args.device)
        freeze_eval(fed)
        self.inl = None
        if inl_checkpoint is not None:
            self.inl = INL().to(args.device)
            load_fin_checkpoint(inl_checkpoint, self.inl, args.device)
            freeze_eval(self.inl)

        super().__init__(method_name or "FIN", fed, target_value=0.0, threshold=0.0)
        self.noise_type = noise_type
        self.fed_checkpoint = str(fed_checkpoint)
        self.inl_checkpoint = str(inl_checkpoint) if inl_checkpoint is not None else None
        self.message_length = 64

    def decode(self, image):
        image = resize_if_needed(image, 128, 128)
        if self.inl is not None:
            image = self.inl(image, rev=True)
        all_zero = torch.zeros(image.shape[0], self.message_length, device=image.device)
        _, decoded = self.model([image, all_zero], rev=True)
        return decoded


class TrustMarkDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["trustmark"])
        add_sys_path(AGENT_ROOT / "trustmark/python")
        from trustmark import TrustMark

        encoding_map = {
            "BCH_5": TrustMark.Encoding.BCH_5,
            "BCH_4": TrustMark.Encoding.BCH_4,
            "BCH_3": TrustMark.Encoding.BCH_3,
            "BCH_SUPER": TrustMark.Encoding.BCH_SUPER,
        }
        tm = TrustMark(
            verbose=False,
            model_type=args.trustmark_model_type,
            encoding_type=encoding_map[args.trustmark_encoding_type],
            loadRemover=False,
            loadBBoxDetector=False,
            device=args.device,
        )
        freeze_eval(tm.encoder)
        freeze_eval(tm.decoder)
        super().__init__("trustmark", tm, target_value=0.5, threshold=0.0)
        self.height = tm.model_resolution_dec
        self.width = tm.model_resolution_dec

    def decode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        return self.model.decoder.decoder(image)

    def clean_bit_confidence(self, decoded):
        return (torch.sigmoid(decoded.detach()) - 0.5).abs()

    def probability_values(self, decoded):
        return torch.sigmoid(decoded).clamp(0.0, 1.0)


class InvisMarkDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["model", "noise", "metrics", "train", "utils"])
        add_sys_path(AGENT_ROOT / "InvisMark")
        import model as invismark_model

        state_dict = torch.load(args.invismark_ckpt, map_location=args.device, weights_only=False)
        # Decoder evaluation does not need InvisMark's training-only Watermark
        # container.  Constructing that container eagerly creates a Noiser,
        # whose v2.JPEG transform is unavailable in torchvision 0.17.  Loading
        # the extractor directly also avoids allocating the unused encoder,
        # discriminator, optimizers, and perceptual-loss modules.
        decoder = invismark_model.Extractor(state_dict["config"]).to(args.device)
        decoder.load_state_dict(state_dict["decoder_state_dict"])
        freeze_eval(decoder)
        super().__init__("invismark", decoder, target_value=0.5, threshold=0.5)

    def decode(self, image):
        return self.model(image)


class RoSteALSDecoder(DecoderWrapper):
    def __init__(self, args):
        purge_modules(["ldm", "cldm", "taming", "taming_transformers", "tools"])
        add_sys_path(AGENT_ROOT / "RoSteALS")
        add_sys_path(AGENT_ROOT / "RoSteALS/taming-transformers-master")
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf

        config = OmegaConf.load(args.rosteals_config).model
        secret_len = config.params.control_config.params.secret_len
        config.params.decoder_config.params.secret_len = secret_len
        model = instantiate_from_config(config)

        state_dict = torch.load(args.rosteals_weight, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        freeze_eval(model.to(args.device))
        super().__init__("rosteals", model, target_value=0.5, threshold=0.0)
        self.image_size = args.rosteals_image_size

    def decode(self, image):
        image = resize_if_needed(image, self.image_size, self.image_size)
        return self.model.decoder(image)

    def clean_bit_confidence(self, decoded):
        return (torch.sigmoid(decoded.detach()) - 0.5).abs()

    def probability_values(self, decoded):
        return torch.sigmoid(decoded).clamp(0.0, 1.0)


def load_lightweightmark_model(args):
    purge_modules(["models", "block", "utils", "config"])
    remove_sys_path(AGENT_ROOT / "FIN")
    remove_sys_path(AGENT_ROOT / "MBRS")
    remove_sys_path(AGENT_ROOT / "HiDDeN")
    remove_sys_path(AGENT_ROOT / "CIN/codes")
    remove_sys_path(AGENT_ROOT / "PIMoG")
    remove_sys_path(AGENT_ROOT / "InvisMark")
    remove_sys_path(AGENT_ROOT / "RoSteALS")
    remove_sys_path(AGENT_ROOT / "trustmark/python")
    add_sys_path(AGENT_ROOT / "LightweightMark")
    from models.Model import PM

    model = PM(args.lightweightmark_mode)
    state_dict = torch.load(args.lightweightmark_model_path, map_location=args.device)
    # PH checkpoints contain the trained ADD/MUT projection stack.  A non-strict
    # load can silently leave those layers randomly initialized when source and
    # checkpoint versions disagree, producing severely inflated logits/gradients.
    model.load_state_dict(state_dict, strict=True)
    freeze_eval(model.to(args.device))
    return model


class LightweightMarkDecoder(DecoderWrapper):
    def __init__(self, args):
        model = load_lightweightmark_model(args)
        super().__init__("lightweightmark", model, target_value=0.0, threshold=0.0)
        self.height = args.lightweightmark_h
        self.width = args.lightweightmark_w

    def decode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        return self.model.decoder(image)


def videoseal_config_for_method(args, method_name):
    if method_name == "chunkyseal":
        return {
            "model_name": getattr(args, "chunkyseal_model_name", "chunkyseal"),
            "message_length": getattr(args, "chunkyseal_message_length", None),
            "short_edge": getattr(args, "chunkyseal_short_edge", 512),
            "lowres_attenuation": getattr(args, "chunkyseal_lowres_attenuation", False),
        }
    return {
        "model_name": args.videoseal_model_name,
        "message_length": args.videoseal_message_length,
        "short_edge": args.videoseal_short_edge,
        "lowres_attenuation": args.videoseal_lowres_attenuation,
    }


def load_videoseal_model(args, method_name="videoseal"):
    purge_modules(["videoseal"])
    add_sys_path(VIDEOSEAL_ROOT)
    import videoseal

    config = videoseal_config_for_method(args, method_name)
    # VideoSeal model cards contain project-relative paths such as
    # configs/attenuation.yaml, and its checkpoint cache is also cwd-relative.
    # Load from the VideoSeal project root, then always restore the caller's cwd.
    previous_cwd = Path.cwd()
    try:
        os.chdir(VIDEOSEAL_ROOT)
        model = videoseal.load(config["model_name"])
    finally:
        os.chdir(previous_cwd)
    freeze_eval(model.to(args.device))
    return model


def get_videoseal_message_length(model):
    if hasattr(model, "embedder") and hasattr(model.embedder, "msg_processor"):
        nbits = getattr(model.embedder.msg_processor, "nbits", None)
        if nbits is not None:
            return int(nbits)
    with torch.no_grad():
        message = model.get_random_msg(1)
    return int(message.shape[1])


class VideoSealDecoder(DecoderWrapper):
    def __init__(self, args, method_name="videoseal"):
        config = videoseal_config_for_method(args, method_name)
        model = load_videoseal_model(args, method_name)
        super().__init__(method_name, model, target_value=0.0, threshold=0.0)
        self.short_edge = config["short_edge"]

    def decode(self, image):
        image_01 = neg1_to_01(image).clamp(0.0, 1.0)
        if self.short_edge and self.short_edge > 0:
            image_01 = F.interpolate(image_01, size=(self.short_edge, self.short_edge), mode="bilinear", align_corners=False)
        # VideoSeal's public detect() API is decorated with no_grad because it
        # is intended for inference. Call the detector directly so a top-1
        # adversarial attack can still differentiate through its logits.
        model_size = int(getattr(self.model, "img_size", image_01.shape[-1]))
        if image_01.shape[-2:] != (model_size, model_size):
            image_01 = F.interpolate(
                image_01,
                size=(model_size, model_size),
                mode="bilinear",
                align_corners=False,
            )
        preds = self.model.detector(image_01.to(self.model.device)).to(image.device)
        return preds[:, 1:]


class EncoderWrapper:
    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    def encode(self, image):
        raise NotImplementedError


class MBRSEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["network", "utils"])
        remove_sys_path(AGENT_ROOT / "HiDDeN")
        remove_sys_path(AGENT_ROOT / "PIMoG")
        remove_sys_path(AGENT_ROOT / "InvisMark")
        remove_sys_path(AGENT_ROOT / "CIN/codes")
        remove_sys_path(AGENT_ROOT / "FIN")
        remove_sys_path(AGENT_ROOT / "trustmark/python")
        remove_sys_path(AGENT_ROOT / "RoSteALS")
        remove_sys_path(AGENT_ROOT / "RoSteALS/taming-transformers-master")
        add_sys_path(AGENT_ROOT / "MBRS")
        from network.Encoder_MP_Decoder import EncoderDecoder, EncoderDecoder_Diffusion

        noise_layers = args.mbrs_noise_layer.copy() if args.mbrs_noise_layer else ["Identity()"]
        if args.mbrs_with_diffusion:
            model = EncoderDecoder_Diffusion(args.mbrs_h, args.mbrs_w, args.mbrs_message_length, noise_layers)
        else:
            model = EncoderDecoder(args.mbrs_h, args.mbrs_w, args.mbrs_message_length, noise_layers)
        state_dict = torch.load(args.mbrs_model_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        freeze_eval(model.to(args.device))
        super().__init__("MBRS", model)
        self.message_length = args.mbrs_message_length
        self.height = args.mbrs_h
        self.width = args.mbrs_w

    def encode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        message = alternating_bits(self.message_length, image.device, dtype=image.dtype)
        return self.model.encoder(image, message).clamp(-1.0, 1.0)


class HiddenEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["model", "models", "noise_layers", "utils"])
        remove_sys_path(AGENT_ROOT / "PIMoG")
        remove_sys_path(AGENT_ROOT / "InvisMark")
        add_sys_path(AGENT_ROOT / "HiDDeN")
        from model.hidden import Hidden
        from noise_layers.noiser import Noiser
        import utils

        _, hidden_config, noise_config = utils.load_options(args.hidden_options_file)
        noiser = Noiser(noise_config, args.device)
        checkpoint = torch.load(args.hidden_checkpoint_file, map_location=args.device)
        model = Hidden(hidden_config, args.device, noiser, None)
        utils.model_from_checkpoint(model, checkpoint)
        freeze_eval(model.encoder_decoder.to(args.device))
        super().__init__("hidden", model.encoder_decoder)
        self.message_length = hidden_config.message_length
        self.height = hidden_config.H
        self.width = hidden_config.W

    def encode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        message = alternating_bits(self.message_length, image.device, dtype=image.dtype)
        encoded, _, _ = self.model(image, message)
        return encoded.clamp(-1.0, 1.0)


class PIMoGEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["model", "Noise_Layer"])
        add_sys_path(AGENT_ROOT / "PIMoG")
        from model import Encoder_Decoder

        model = Encoder_Decoder(args.pimog_distortion).to(args.device)
        checkpoint_path = (
            Path(args.pimog_model_save_dir)
            / args.pimog_distortion
            / f"{args.pimog_model_name}_mask_{args.pimog_embedding_epoch}.pth"
        )
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        freeze_eval(model)
        super().__init__("PIMoG", model)
        self.image_size = args.pimog_image_size
        self.message_length = 30

    def encode(self, image):
        image = resize_if_needed(image, self.image_size, self.image_size)
        bgr = image[:, [2, 1, 0], :, :]
        message = alternating_bits(self.message_length, image.device, dtype=image.dtype)
        encoded_bgr, _, _ = self.model(bgr.float(), message)
        encoded_rgb = encoded_bgr[:, [2, 1, 0], :, :]
        return encoded_rgb.clamp(-1.0, 1.0)


class StegaStampEncoder(EncoderWrapper):
    def __init__(self, args):
        model = load_stegastamp_component(args, "encoder")
        super().__init__("stegastamp", model)
        self.image_size = args.stegastamp_image_size
        self.message_length = args.stegastamp_message_length

    def encode(self, image):
        image = resize_if_needed(image, self.image_size, self.image_size)
        image_01 = ((image + 1.0) * 0.5).clamp(0.0, 1.0)
        message = alternating_bits(
            self.message_length,
            image.device,
            dtype=image.dtype,
        )
        residual = self.model(message, image_01)
        encoded_01 = (image_01 + residual).clamp(0.0, 1.0)
        return (encoded_01 * 2.0 - 1.0).clamp(-1.0, 1.0)


class CINEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["models", "utils"])
        add_sys_path(AGENT_ROOT / "CIN/codes")
        from models.CIN import CIN
        from utils.yml import dict_to_nonedict, parse_yml

        option_yml = parse_yml(args.cin_options_file)
        opt = dict_to_nonedict(option_yml)
        opt["train"]["batch_size"] = 1
        opt["train"]["num_workers"] = 0
        opt["path"]["folder_temp"] = str(Path(args.output_dir).resolve() / "cin_temp")
        model = CIN(opt, args.device).to(args.device)
        checkpoint = torch.load(args.cin_checkpoint, map_location=args.device)
        state_dict = checkpoint.get("cinNet", checkpoint)
        current_state = model.state_dict()
        loaded_state = {}
        for key, value in state_dict.items():
            target_key = checkpoint_target_key(key, current_state)
            if target_key is not None:
                loaded_state[target_key] = value
        current_state.update(loaded_state)
        model.load_state_dict(current_state, strict=False)
        freeze_eval(model)
        super().__init__("CIN", model)
        self.height = opt["datasets"]["H"]
        self.width = opt["datasets"]["W"]
        self.message_length = opt["network"]["message_length"]

    def encode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        message = alternating_bits(self.message_length, image.device, dtype=image.dtype)
        return self.model.encoder(image, message).clamp(-1.0, 1.0)


class FINEncoder(EncoderWrapper):
    def __init__(self, args, noise_type=None, method_name=None):
        prepare_fin_import_path()
        from models.encoder_decoder import FED

        noise_type = (noise_type or args.fin_noise_type).upper()
        if noise_type not in {"HEAVY", "JPEG"}:
            raise ValueError(f"Unsupported FIN noise type: {noise_type}")
        explicit_variant = method_name is not None
        variant_checkpoint = getattr(args, f"fin_{noise_type.lower()}_fed_checkpoint", None)
        generic_checkpoint = None if explicit_variant else args.fin_fed_checkpoint
        checkpoint_value = variant_checkpoint or generic_checkpoint
        fed_checkpoint = Path(checkpoint_value) if checkpoint_value else (
            AGENT_ROOT / "FIN/experiments" / noise_type / "FED.pt"
        )
        fed = FED().to(args.device)
        load_fin_checkpoint(fed_checkpoint, fed, args.device)
        freeze_eval(fed)
        super().__init__(method_name or "FIN", fed)
        self.noise_type = noise_type
        self.fed_checkpoint = str(fed_checkpoint)
        self.message_length = 64

    def encode(self, image):
        image = resize_if_needed(image, 128, 128)
        message = alternating_bits(self.message_length, image.device, dtype=image.dtype) - 0.5
        stego, _ = self.model([image, message])
        return stego.clamp(-1.0, 1.0)


class TrustMarkEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["trustmark"])
        add_sys_path(AGENT_ROOT / "trustmark/python")
        from trustmark import TrustMark

        encoding_map = {
            "BCH_5": TrustMark.Encoding.BCH_5,
            "BCH_4": TrustMark.Encoding.BCH_4,
            "BCH_3": TrustMark.Encoding.BCH_3,
            "BCH_SUPER": TrustMark.Encoding.BCH_SUPER,
        }
        tm = TrustMark(
            verbose=False,
            model_type=args.trustmark_model_type,
            encoding_type=encoding_map[args.trustmark_encoding_type],
            loadRemover=False,
            loadBBoxDetector=False,
            device=args.device,
        )
        freeze_eval(tm.encoder)
        super().__init__("trustmark", tm)
        self.capacity = tm.schemaCapacity()

    def encode(self, image):
        image_01 = ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).squeeze(0)
        image_pil = transforms.ToPILImage()(image_01)
        bits = alternating_bits(self.capacity, torch.device("cpu"), dtype=torch.float32)
        secret = "".join(str(int(bit)) for bit in bits.squeeze(0).tolist())
        encoded = self.model.encode(image_pil.convert("RGB"), secret, MODE="binary", WM_STRENGTH=1.0)
        tensor = transforms.ToTensor()(encoded).unsqueeze(0).to(device=image.device, dtype=image.dtype)
        return (tensor * 2.0 - 1.0).clamp(-1.0, 1.0)


class InvisMarkEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["model", "noise", "metrics", "train", "utils"])
        add_sys_path(AGENT_ROOT / "InvisMark")
        import model as invismark_model

        state_dict = torch.load(args.invismark_ckpt, map_location=args.device, weights_only=False)
        config = state_dict["config"]
        encoder = invismark_model.Encoder(config).to(args.device)
        encoder.load_state_dict(state_dict["encoder_state_dict"], strict=True)
        freeze_eval(encoder)
        super().__init__("invismark", encoder)
        self.message_length = config.num_encoded_bits
        self.image_shape = tuple(config.image_shape)

    def encode(self, image):
        message = alternating_bits(self.message_length, image.device, dtype=image.dtype)
        resized = transforms.Resize(self.image_shape)(image)
        encoded_resized = self.model(resized, message)
        residual = transforms.Resize(image.shape[-2:])(encoded_resized - resized)
        return (image + residual).clamp(-1.0, 1.0)


class RoSteALSEncoder(EncoderWrapper):
    def __init__(self, args):
        purge_modules(["ldm", "cldm", "taming", "taming_transformers", "tools"])
        add_sys_path(AGENT_ROOT / "RoSteALS")
        add_sys_path(AGENT_ROOT / "RoSteALS/taming-transformers-master")
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf

        config = OmegaConf.load(args.rosteals_config).model
        secret_len = config.params.control_config.params.secret_len
        config.params.decoder_config.params.secret_len = secret_len
        model = instantiate_from_config(config)
        state_dict = torch.load(args.rosteals_weight, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        freeze_eval(model.to(args.device))
        super().__init__("rosteals", model)
        self.message_length = secret_len
        self.image_size = args.rosteals_image_size

    def encode(self, image):
        image = resize_if_needed(image, self.image_size, self.image_size)
        secret = alternating_bits(self.message_length, image.device, dtype=image.dtype)
        z = self.model.encode_first_stage(image)
        z_embed, _ = self.model(z, None, secret)
        stego = self.model.decode_first_stage(z_embed)
        return stego.clamp(-1.0, 1.0)


class LightweightMarkEncoder(EncoderWrapper):
    def __init__(self, args):
        model = load_lightweightmark_model(args)
        super().__init__("lightweightmark", model)
        self.message_length = args.lightweightmark_message_length
        self.height = args.lightweightmark_h
        self.width = args.lightweightmark_w
        self.device = args.device

    def encode(self, image):
        image = resize_if_needed(image, self.height, self.width)
        message = alternating_centered_bits(self.message_length, image.device, dtype=image.dtype)
        return self.model.encoder(image, message).clamp(-1.0, 1.0)


class VideoSealEncoder(EncoderWrapper):
    def __init__(self, args, method_name="videoseal"):
        config = videoseal_config_for_method(args, method_name)
        model = load_videoseal_model(args, method_name)
        super().__init__(method_name, model)
        self.message_length = config["message_length"] or get_videoseal_message_length(model)
        self.short_edge = config["short_edge"]
        self.lowres_attenuation = config["lowres_attenuation"]
        self.device = args.device

    def encode(self, image):
        image_01 = neg1_to_01(image).clamp(0.0, 1.0)
        if self.short_edge and self.short_edge > 0:
            image_01 = F.interpolate(image_01, size=(self.short_edge, self.short_edge), mode="bilinear", align_corners=False)
        message = alternating_bits(self.message_length, image_01.device, dtype=image_01.dtype)
        outputs = self.model.embed(
            image_01,
            msgs=message,
            is_video=False,
            lowres_attenuation=self.lowres_attenuation,
        )
        return (outputs["imgs_w"].clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)


def alternating_bits(length, device, dtype=torch.float32):
    values = torch.arange(length, device=device).remainder(2).to(dtype=dtype)
    return values.unsqueeze(0)


def alternating_centered_bits(length, device, dtype=torch.float32):
    return alternating_bits(length, device, dtype=dtype) * 2.0 - 1.0


def canonical_method(name):
    normalized = name.strip()
    lower = normalized.lower()
    if lower == "pimog":
        return "pimog"
    if lower == "hidden":
        return "hidden"
    if lower == "mbrs":
        return "mbrs"
    if lower in {"stegastamp", "stega_stamp"}:
        return "stegastamp"
    if lower == "invismark":
        return "invismark"
    if lower == "rosteals":
        return "rosteals"
    if lower in {"fin_heavy", "fin-heavy", "finheavy"}:
        return "fin_heavy"
    if lower in {"fin_jpeg", "fin-jpeg", "finjpeg"}:
        return "fin_jpeg"
    if lower in {"lightweightmark", "lightweight_mark", "lightweight"}:
        return "lightweightmark"
    if lower in {"videoseal", "video_seal"}:
        return "videoseal"
    if lower in {"chunkyseal", "chunky_seal", "cunkyseal"}:
        return "chunkyseal"
    return lower




def build_decoder(method, args):
    method = canonical_method(method)
    if method == "mbrs":
        return MBRSDecoder(args)
    if method == "hidden":
        return HiddenDecoder(args)
    if method == "pimog":
        return PIMoGDecoder(args)
    if method == "stegastamp":
        return StegaStampDecoder(args)
    if method == "cin":
        return CINDecoder(args)
    if method == "fin":
        return FINDecoder(args)
    if method == "fin_heavy":
        return FINDecoder(args, noise_type="HEAVY", method_name="FIN_HEAVY")
    if method == "fin_jpeg":
        return FINDecoder(args, noise_type="JPEG", method_name="FIN_JPEG")
    if method == "trustmark":
        return TrustMarkDecoder(args)
    if method == "invismark":
        return InvisMarkDecoder(args)
    if method == "rosteals":
        return RoSteALSDecoder(args)
    if method == "lightweightmark":
        return LightweightMarkDecoder(args)
    if method == "videoseal":
        return VideoSealDecoder(args)
    if method == "chunkyseal":
        return VideoSealDecoder(args, method_name="chunkyseal")
    raise ValueError(f"No differentiable decoder wrapper for method: {method}")


def build_encoder(method, args):
    method = canonical_method(method)
    if method == "mbrs":
        return MBRSEncoder(args)
    if method == "hidden":
        return HiddenEncoder(args)
    if method == "pimog":
        return PIMoGEncoder(args)
    if method == "stegastamp":
        return StegaStampEncoder(args)
    if method == "cin":
        return CINEncoder(args)
    if method == "fin":
        return FINEncoder(args)
    if method == "fin_heavy":
        return FINEncoder(args, noise_type="HEAVY", method_name="FIN_HEAVY")
    if method == "fin_jpeg":
        return FINEncoder(args, noise_type="JPEG", method_name="FIN_JPEG")
    if method == "trustmark":
        return TrustMarkEncoder(args)
    if method == "invismark":
        return InvisMarkEncoder(args)
    if method == "rosteals":
        return RoSteALSEncoder(args)
    if method == "lightweightmark":
        return LightweightMarkEncoder(args)
    if method == "videoseal":
        return VideoSealEncoder(args)
    if method == "chunkyseal":
        return VideoSealEncoder(args, method_name="chunkyseal")
    raise ValueError(f"No encoder wrapper for method: {method}")


def load_image_tensor(path, device, input_size=None):
    image = Image.open(path).convert("RGB")
    ops = []
    if input_size is not None:
        ops.append(transforms.Resize((input_size, input_size)))
    ops.extend([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    tensor = transforms.Compose(ops)(image).unsqueeze(0).to(device)
    return tensor, image.size


def tensor_image_to_uint8(image_tensor):
    image = image_tensor.detach().cpu().clamp(-1.0, 1.0)
    return ((image + 1.0) * 127.5).round().byte().squeeze(0).permute(1, 2, 0).numpy()


def add_decoded_margin_diagnostics(decoder, decoded, metric):
    probabilities = decoder.probability_values(decoded.detach())
    margin = (probabilities - 0.5).abs()
    metric.update(
        {
            "decoded_margin_to_0_5_mean": float(margin.mean().item()),
            "decoded_margin_to_0_5_min": float(margin.min().item()),
            "decoded_margin_to_0_5_max": float(margin.max().item()),
        }
    )
    if decoder.clean_decoded is None:
        return
    clean_probabilities = decoder.probability_values(decoder.clean_decoded.detach())
    compare_len = min(probabilities.shape[1], clean_probabilities.shape[1])
    probabilities = probabilities[:, :compare_len]
    clean_probabilities = clean_probabilities[:, :compare_len]
    clean_margin = (clean_probabilities - 0.5).abs()
    decoded_delta = (probabilities - clean_probabilities).abs()
    metric.update(
        {
            "clean_decoded_margin_to_0_5_mean": float(clean_margin.mean().item()),
            "clean_decoded_margin_to_0_5_min": float(clean_margin.min().item()),
            "clean_decoded_margin_to_0_5_max": float(clean_margin.max().item()),
            "decoded_delta_from_clean_mean": float(decoded_delta.mean().item()),
            "decoded_delta_from_clean_max": float(decoded_delta.max().item()),
        }
    )
