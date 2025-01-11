# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.47.0.dev0"

from typing import TYPE_CHECKING

# Check the dependencies satisfy the minimal versions required.
from transformers.src.transformers import dependency_versions_check #!Changed
from transformers.src.transformers.utils import ( #!Changed
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_bitsandbytes_available,
    is_essentia_available,
    is_flax_available,
    is_g2p_en_available,
    is_keras_nlp_available,
    is_librosa_available,
    is_pretty_midi_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_speech_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torchaudio_available,
    is_torchvision_available,
    is_vision_available,
    logging,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Base objects, independent of any specific backend
_import_structure = {
    "agents": [
        "Agent",
        "CodeAgent",
        "HfApiEngine",
        "ManagedAgent",
        "PipelineTool",
        "ReactAgent",
        "ReactCodeAgent",
        "ReactJsonAgent",
        "Tool",
        "Toolbox",
        "ToolCollection",
        "TransformersEngine",
        "launch_gradio_demo",
        "load_tool",
        "stream_to_gradio",
        "tool",
    ],
    "audio_utils": [],
    "benchmark": [],
    "commands": [],
    "configuration_utils": ["PretrainedConfig"],
    "convert_graph_to_onnx": [],
    "convert_slow_tokenizers_checkpoints_to_fast": [],
    "convert_tf_hub_seq_to_seq_bert_to_pytorch": [],
    "data": [
        "DataProcessor",
        "InputExample",
        "InputFeatures",
        "SingleSentenceClassificationProcessor",
        "SquadExample",
        "SquadFeatures",
        "SquadV1Processor",
        "SquadV2Processor",
        "glue_compute_metrics",
        "glue_convert_examples_to_features",
        "glue_output_modes",
        "glue_processors",
        "glue_tasks_num_labels",
        "squad_convert_examples_to_features",
        "xnli_compute_metrics",
        "xnli_output_modes",
        "xnli_processors",
        "xnli_tasks_num_labels",
    ],
    "data.data_collator": [
        "DataCollator",
        "DataCollatorForLanguageModeling",
        "DataCollatorForPermutationLanguageModeling",
        "DataCollatorForSeq2Seq",
        "DataCollatorForSOP",
        "DataCollatorForTokenClassification",
        "DataCollatorForWholeWordMask",
        "DataCollatorWithFlattening",
        "DataCollatorWithPadding",
        "DefaultDataCollator",
        "default_data_collator",
    ],
    "data.metrics": [],
    "data.processors": [],
    "debug_utils": [],
    "dependency_versions_check": [],
    "dependency_versions_table": [],
    "dynamic_module_utils": [],
    "feature_extraction_sequence_utils": ["SequenceFeatureExtractor"],
    "feature_extraction_utils": ["BatchFeature", "FeatureExtractionMixin"],
    "file_utils": [],
    "generation": [
        "GenerationConfig",
        "TextIteratorStreamer",
        "TextStreamer",
        "WatermarkingConfig",
    ],
    "hf_argparser": ["HfArgumentParser"],
    "hyperparameter_search": [],
    "image_transforms": [],
    "integrations": [
        "is_clearml_available",
        "is_comet_available",
        "is_dvclive_available",
        "is_neptune_available",
        "is_optuna_available",
        "is_ray_available",
        "is_ray_tune_available",
        "is_sigopt_available",
        "is_tensorboard_available",
        "is_wandb_available",
    ],
    "loss": [],
    "modelcard": ["ModelCard"],
    # Losses
    "modeling_tf_pytorch_utils": [
        "convert_tf_weight_name_to_pt_weight_name",
        "load_pytorch_checkpoint_in_tf2_model",
        "load_pytorch_model_in_tf2_model",
        "load_pytorch_weights_in_tf2_model",
        "load_tf2_checkpoint_in_pytorch_model",
        "load_tf2_model_in_pytorch_model",
        "load_tf2_weights_in_pytorch_model",
    ],
    # Models
    "models": [],
    "models.albert": ["AlbertConfig"],
    "models.align": [
        "AlignConfig",
        "AlignProcessor",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "models.altclip": [
        "AltCLIPConfig",
        "AltCLIPProcessor",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "models.audio_spectrogram_transformer": [
        "ASTConfig",
        "ASTFeatureExtractor",
    ],
    "models.auto": [
        "CONFIG_MAPPING",
        "FEATURE_EXTRACTOR_MAPPING",
        "IMAGE_PROCESSOR_MAPPING",
        "MODEL_NAMES_MAPPING",
        "PROCESSOR_MAPPING",
        "TOKENIZER_MAPPING",
        "AutoConfig",
        "AutoFeatureExtractor",
        "AutoImageProcessor",
        "AutoProcessor",
        "AutoTokenizer",
    ],
    "models.autoformer": ["AutoformerConfig"],
    "models.bark": [
        "BarkCoarseConfig",
        "BarkConfig",
        "BarkFineConfig",
        "BarkProcessor",
        "BarkSemanticConfig",
    ],
    "models.bart": ["BartConfig", "BartTokenizer"],
    "models.barthez": [],
    "models.bartpho": [],
    "models.beit": ["BeitConfig"],
    "models.bert": [
        "BasicTokenizer",
        "BertConfig",
        "BertTokenizer",
        "WordpieceTokenizer",
    ],
    "models.bert_generation": ["BertGenerationConfig"],
    "models.bert_japanese": [
        "BertJapaneseTokenizer",
        "CharacterTokenizer",
        "MecabTokenizer",
    ],
    "models.bertweet": ["BertweetTokenizer"],
    "models.big_bird": ["BigBirdConfig"],
    "models.bigbird_pegasus": ["BigBirdPegasusConfig"],
    "models.biogpt": [
        "BioGptConfig",
        "BioGptTokenizer",
    ],
    "models.bit": ["BitConfig"],
    "models.blenderbot": [
        "BlenderbotConfig",
        "BlenderbotTokenizer",
    ],
    "models.blenderbot_small": [
        "BlenderbotSmallConfig",
        "BlenderbotSmallTokenizer",
    ],
    "models.blip": [
        "BlipConfig",
        "BlipProcessor",
        "BlipTextConfig",
        "BlipVisionConfig",
    ],
    "models.blip_2": [
        "Blip2Config",
        "Blip2Processor",
        "Blip2QFormerConfig",
        "Blip2VisionConfig",
    ],
    "models.bloom": ["BloomConfig"],
    "models.bridgetower": [
        "BridgeTowerConfig",
        "BridgeTowerProcessor",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "models.bros": [
        "BrosConfig",
        "BrosProcessor",
    ],
    "models.byt5": ["ByT5Tokenizer"],
    "models.camembert": ["CamembertConfig"],
    "models.canine": [
        "CanineConfig",
        "CanineTokenizer",
    ],
    "models.chameleon": [
        "ChameleonConfig",
        "ChameleonProcessor",
        "ChameleonVQVAEConfig",
    ],
    "models.chinese_clip": [
        "ChineseCLIPConfig",
        "ChineseCLIPProcessor",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "models.clap": [
        "ClapAudioConfig",
        "ClapConfig",
        "ClapProcessor",
        "ClapTextConfig",
    ],
    "models.clip": [
        "CLIPConfig",
        "CLIPProcessor",
        "CLIPTextConfig",
        "CLIPTokenizer",
        "CLIPVisionConfig",
    ],
    "models.clipseg": [
        "CLIPSegConfig",
        "CLIPSegProcessor",
        "CLIPSegTextConfig",
        "CLIPSegVisionConfig",
    ],
    "models.clvp": [
        "ClvpConfig",
        "ClvpDecoderConfig",
        "ClvpEncoderConfig",
        "ClvpFeatureExtractor",
        "ClvpProcessor",
        "ClvpTokenizer",
    ],
    "models.code_llama": [],
    "models.codegen": [
        "CodeGenConfig",
        "CodeGenTokenizer",
    ],
    "models.cohere": ["CohereConfig"],
    "models.conditional_detr": ["ConditionalDetrConfig"],
    "models.convbert": [
        "ConvBertConfig",
        "ConvBertTokenizer",
    ],
    "models.convnext": ["ConvNextConfig"],
    "models.convnextv2": ["ConvNextV2Config"],
    "models.cpm": [],
    "models.cpmant": [
        "CpmAntConfig",
        "CpmAntTokenizer",
    ],
    "models.ctrl": [
        "CTRLConfig",
        "CTRLTokenizer",
    ],
    "models.cvt": ["CvtConfig"],
    "models.dac": ["DacConfig", "DacFeatureExtractor"],
    "models.data2vec": [
        "Data2VecAudioConfig",
        "Data2VecTextConfig",
        "Data2VecVisionConfig",
    ],
    "models.dbrx": ["DbrxConfig"],
    "models.deberta": [
        "DebertaConfig",
        "DebertaTokenizer",
    ],
    "models.deberta_v2": ["DebertaV2Config"],
    "models.decision_transformer": ["DecisionTransformerConfig"],
    "models.deformable_detr": ["DeformableDetrConfig"],
    "models.deit": ["DeiTConfig"],
    "models.deprecated": [],
    "models.deprecated.bort": [],
    "models.deprecated.deta": ["DetaConfig"],
    "models.deprecated.efficientformer": ["EfficientFormerConfig"],
    "models.deprecated.ernie_m": ["ErnieMConfig"],
    "models.deprecated.gptsan_japanese": [
        "GPTSanJapaneseConfig",
        "GPTSanJapaneseTokenizer",
    ],
    "models.deprecated.graphormer": ["GraphormerConfig"],
    "models.deprecated.jukebox": [
        "JukeboxConfig",
        "JukeboxPriorConfig",
        "JukeboxTokenizer",
        "JukeboxVQVAEConfig",
    ],
    "models.deprecated.mctct": [
        "MCTCTConfig",
        "MCTCTFeatureExtractor",
        "MCTCTProcessor",
    ],
    "models.deprecated.mega": ["MegaConfig"],
    "models.deprecated.mmbt": ["MMBTConfig"],
    "models.deprecated.nat": ["NatConfig"],
    "models.deprecated.nezha": ["NezhaConfig"],
    "models.deprecated.open_llama": ["OpenLlamaConfig"],
    "models.deprecated.qdqbert": ["QDQBertConfig"],
    "models.deprecated.realm": [
        "RealmConfig",
        "RealmTokenizer",
    ],
    "models.deprecated.retribert": [
        "RetriBertConfig",
        "RetriBertTokenizer",
    ],
    "models.deprecated.speech_to_text_2": [
        "Speech2Text2Config",
        "Speech2Text2Processor",
        "Speech2Text2Tokenizer",
    ],
    "models.deprecated.tapex": ["TapexTokenizer"],
    "models.deprecated.trajectory_transformer": ["TrajectoryTransformerConfig"],
    "models.deprecated.transfo_xl": [
        "TransfoXLConfig",
        "TransfoXLCorpus",
        "TransfoXLTokenizer",
    ],
    "models.deprecated.tvlt": [
        "TvltConfig",
        "TvltFeatureExtractor",
        "TvltProcessor",
    ],
    "models.deprecated.van": ["VanConfig"],
    "models.deprecated.vit_hybrid": ["ViTHybridConfig"],
    "models.deprecated.xlm_prophetnet": ["XLMProphetNetConfig"],
    "models.depth_anything": ["DepthAnythingConfig"],
    "models.detr": ["DetrConfig"],
    "models.dialogpt": [],
    "models.dinat": ["DinatConfig"],
    "models.dinov2": ["Dinov2Config"],
    "models.distilbert": [
        "DistilBertConfig",
        "DistilBertTokenizer",
    ],
    "models.dit": [],
    "models.donut": [
        "DonutProcessor",
        "DonutSwinConfig",
    ],
    "models.dpr": [
        "DPRConfig",
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
    "models.dpt": ["DPTConfig"],
    "models.efficientnet": ["EfficientNetConfig"],
    "models.electra": [
        "ElectraConfig",
        "ElectraTokenizer",
    ],
    "models.encodec": [
        "EncodecConfig",
        "EncodecFeatureExtractor",
    ],
    "models.encoder_decoder": ["EncoderDecoderConfig"],
    "models.ernie": ["ErnieConfig"],
    "models.esm": ["EsmConfig", "EsmTokenizer"],
    "models.falcon": ["FalconConfig"],
    "models.falcon_mamba": ["FalconMambaConfig"],
    "models.fastspeech2_conformer": [
        "FastSpeech2ConformerConfig",
        "FastSpeech2ConformerHifiGanConfig",
        "FastSpeech2ConformerTokenizer",
        "FastSpeech2ConformerWithHifiGanConfig",
    ],
    "models.flaubert": ["FlaubertConfig", "FlaubertTokenizer"],
    "models.flava": [
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
    "models.fnet": ["FNetConfig"],
    "models.focalnet": ["FocalNetConfig"],
    "models.fsmt": [
        "FSMTConfig",
        "FSMTTokenizer",
    ],
    "models.funnel": [
        "FunnelConfig",
        "FunnelTokenizer",
    ],
    "models.fuyu": ["FuyuConfig"],
    "models.gemma": ["GemmaConfig"],
    "models.gemma2": ["Gemma2Config"],
    "models.git": [
        "GitConfig",
        "GitProcessor",
        "GitVisionConfig",
    ],
    "models.glm": ["GlmConfig"],
    "models.glpn": ["GLPNConfig"],
    "models.gpt2": [
        "GPT2Config",
        "GPT2Tokenizer",
    ],
    "models.gpt_bigcode": ["GPTBigCodeConfig"],
    "models.gpt_neo": ["GPTNeoConfig"],
    "models.gpt_neox": ["GPTNeoXConfig"],
    "models.gpt_neox_japanese": ["GPTNeoXJapaneseConfig"],
    "models.gpt_sw3": [],
    "models.gptj": ["GPTJConfig"],
    "models.granite": ["GraniteConfig"],
    "models.granitemoe": ["GraniteMoeConfig"],
    "models.grounding_dino": [
        "GroundingDinoConfig",
        "GroundingDinoProcessor",
    ],
    "models.groupvit": [
        "GroupViTConfig",
        "GroupViTTextConfig",
        "GroupViTVisionConfig",
    ],
    "models.herbert": ["HerbertTokenizer"],
    "models.hiera": ["HieraConfig"],
    "models.hubert": ["HubertConfig"],
    "models.ibert": ["IBertConfig"],
    "models.idefics": ["IdeficsConfig"],
    "models.idefics2": ["Idefics2Config"],
    "models.idefics3": ["Idefics3Config"],
    "models.imagegpt": ["ImageGPTConfig"],
    "models.informer": ["InformerConfig"],
    "models.instructblip": [
        "InstructBlipConfig",
        "InstructBlipProcessor",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "models.instructblipvideo": [
        "InstructBlipVideoConfig",
        "InstructBlipVideoProcessor",
        "InstructBlipVideoQFormerConfig",
        "InstructBlipVideoVisionConfig",
    ],
    "models.jamba": ["JambaConfig"],
    "models.jetmoe": ["JetMoeConfig"],
    "models.kosmos2": [
        "Kosmos2Config",
        "Kosmos2Processor",
    ],
    "models.layoutlm": [
        "LayoutLMConfig",
        "LayoutLMTokenizer",
    ],
    "models.layoutlmv2": [
        "LayoutLMv2Config",
        "LayoutLMv2FeatureExtractor",
        "LayoutLMv2ImageProcessor",
        "LayoutLMv2Processor",
        "LayoutLMv2Tokenizer",
    ],
    "models.layoutlmv3": [
        "LayoutLMv3Config",
        "LayoutLMv3FeatureExtractor",
        "LayoutLMv3ImageProcessor",
        "LayoutLMv3Processor",
        "LayoutLMv3Tokenizer",
    ],
    "models.layoutxlm": ["LayoutXLMProcessor"],
    "models.led": ["LEDConfig", "LEDTokenizer"],
    "models.levit": ["LevitConfig"],
    "models.lilt": ["LiltConfig"],
    "models.llama": ["LlamaConfig"],
    "models.llava": [
        "LlavaConfig",
        "LlavaProcessor",
    ],
    "models.llava_next": [
        "LlavaNextConfig",
        "LlavaNextProcessor",
    ],
    "models.llava_next_video": [
        "LlavaNextVideoConfig",
        "LlavaNextVideoProcessor",
    ],
    "models.llava_onevision": ["LlavaOnevisionConfig", "LlavaOnevisionProcessor"],
    "models.longformer": [
        "LongformerConfig",
        "LongformerTokenizer",
    ],
    "models.longt5": ["LongT5Config"],
    "models.luke": [
        "LukeConfig",
        "LukeTokenizer",
    ],
    "models.lxmert": [
        "LxmertConfig",
        "LxmertTokenizer",
    ],
    "models.m2m_100": ["M2M100Config"],
    "models.mamba": ["MambaConfig"],
    "models.mamba2": ["Mamba2Config"],
    "models.marian": ["MarianConfig"],
    "models.markuplm": [
        "MarkupLMConfig",
        "MarkupLMFeatureExtractor",
        "MarkupLMProcessor",
        "MarkupLMTokenizer",
    ],
    "models.mask2former": ["Mask2FormerConfig"],
    "models.maskformer": [
        "MaskFormerConfig",
        "MaskFormerSwinConfig",
    ],
    "models.mbart": ["MBartConfig"],
    "models.mbart50": [],
    "models.megatron_bert": ["MegatronBertConfig"],
    "models.megatron_gpt2": [],
    "models.mgp_str": [
        "MgpstrConfig",
        "MgpstrProcessor",
        "MgpstrTokenizer",
    ],
    "models.mimi": ["MimiConfig"],
    "models.mistral": ["MistralConfig"],
    "models.mixtral": ["MixtralConfig"],
    "models.mllama": [
        "MllamaConfig",
        "MllamaProcessor",
    ],
    "models.mluke": [],
    "models.mobilebert": [
        "MobileBertConfig",
        "MobileBertTokenizer",
    ],
    "models.mobilenet_v1": ["MobileNetV1Config"],
    "models.mobilenet_v2": ["MobileNetV2Config"],
    "models.mobilevit": ["MobileViTConfig"],
    "models.mobilevitv2": ["MobileViTV2Config"],
    "models.moshi": [
        "MoshiConfig",
        "MoshiDepthConfig",
    ],
    "models.mpnet": [
        "MPNetConfig",
        "MPNetTokenizer",
    ],
    "models.mpt": ["MptConfig"],
    "models.mra": ["MraConfig"],
    "models.mt5": ["MT5Config"],
    "models.musicgen": [
        "MusicgenConfig",
        "MusicgenDecoderConfig",
    ],
    "models.musicgen_melody": [
        "MusicgenMelodyConfig",
        "MusicgenMelodyDecoderConfig",
    ],
    "models.mvp": ["MvpConfig", "MvpTokenizer"],
    "models.myt5": ["MyT5Tokenizer"],
    "models.nemotron": ["NemotronConfig"],
    "models.nllb": [],
    "models.nllb_moe": ["NllbMoeConfig"],
    "models.nougat": ["NougatProcessor"],
    "models.nystromformer": ["NystromformerConfig"],
    "models.olmo": ["OlmoConfig"],
    "models.olmoe": ["OlmoeConfig"],
    "models.omdet_turbo": [
        "OmDetTurboConfig",
        "OmDetTurboProcessor",
    ],
    "models.oneformer": [
        "OneFormerConfig",
        "OneFormerProcessor",
    ],
    "models.openai": [
        "OpenAIGPTConfig",
        "OpenAIGPTTokenizer",
    ],
    "models.opt": ["OPTConfig"],
    "models.owlv2": [
        "Owlv2Config",
        "Owlv2Processor",
        "Owlv2TextConfig",
        "Owlv2VisionConfig",
    ],
    "models.owlvit": [
        "OwlViTConfig",
        "OwlViTProcessor",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "models.paligemma": ["PaliGemmaConfig"],
    "models.patchtsmixer": ["PatchTSMixerConfig"],
    "models.patchtst": ["PatchTSTConfig"],
    "models.pegasus": [
        "PegasusConfig",
        "PegasusTokenizer",
    ],
    "models.pegasus_x": ["PegasusXConfig"],
    "models.perceiver": [
        "PerceiverConfig",
        "PerceiverTokenizer",
    ],
    "models.persimmon": ["PersimmonConfig"],
    "models.phi": ["PhiConfig"],
    "models.phi3": ["Phi3Config"],
    "models.phimoe": ["PhimoeConfig"],
    "models.phobert": ["PhobertTokenizer"],
    "models.pix2struct": [
        "Pix2StructConfig",
        "Pix2StructProcessor",
        "Pix2StructTextConfig",
        "Pix2StructVisionConfig",
    ],
    "models.pixtral": ["PixtralProcessor", "PixtralVisionConfig"],
    "models.plbart": ["PLBartConfig"],
    "models.poolformer": ["PoolFormerConfig"],
    "models.pop2piano": ["Pop2PianoConfig"],
    "models.prophetnet": [
        "ProphetNetConfig",
        "ProphetNetTokenizer",
    ],
    "models.pvt": ["PvtConfig"],
    "models.pvt_v2": ["PvtV2Config"],
    "models.qwen2": [
        "Qwen2Config",
        "Qwen2Tokenizer",
    ],
    "models.qwen2_audio": [
        "Qwen2AudioConfig",
        "Qwen2AudioEncoderConfig",
        "Qwen2AudioProcessor",
    ],
    "models.qwen2_moe": ["Qwen2MoeConfig"],
    "models.qwen2_vl": [
        "Qwen2VLConfig",
        "Qwen2VLProcessor",
    ],
    "models.rag": ["RagConfig", "RagRetriever", "RagTokenizer"],
    "models.recurrent_gemma": ["RecurrentGemmaConfig"],
    "models.reformer": ["ReformerConfig"],
    "models.regnet": ["RegNetConfig"],
    "models.rembert": ["RemBertConfig"],
    "models.resnet": ["ResNetConfig"],
    "models.roberta": [
        "RobertaConfig",
        "RobertaTokenizer",
    ],
    "models.roberta_prelayernorm": ["RobertaPreLayerNormConfig"],
    "models.roc_bert": [
        "RoCBertConfig",
        "RoCBertTokenizer",
    ],
    "models.roformer": [
        "RoFormerConfig",
        "RoFormerTokenizer",
    ],
    "models.rt_detr": ["RTDetrConfig", "RTDetrResNetConfig"],
    "models.rwkv": ["RwkvConfig"],
    "models.sam": [
        "SamConfig",
        "SamMaskDecoderConfig",
        "SamProcessor",
        "SamPromptEncoderConfig",
        "SamVisionConfig",
    ],
    "models.seamless_m4t": [
        "SeamlessM4TConfig",
        "SeamlessM4TFeatureExtractor",
        "SeamlessM4TProcessor",
    ],
    "models.seamless_m4t_v2": ["SeamlessM4Tv2Config"],
    "models.segformer": ["SegformerConfig"],
    "models.seggpt": ["SegGptConfig"],
    "models.sew": ["SEWConfig"],
    "models.sew_d": ["SEWDConfig"],
    "models.siglip": [
        "SiglipConfig",
        "SiglipProcessor",
        "SiglipTextConfig",
        "SiglipVisionConfig",
    ],
    "models.speech_encoder_decoder": ["SpeechEncoderDecoderConfig"],
    "models.speech_to_text": [
        "Speech2TextConfig",
        "Speech2TextFeatureExtractor",
        "Speech2TextProcessor",
    ],
    "models.speecht5": [
        "SpeechT5Config",
        "SpeechT5FeatureExtractor",
        "SpeechT5HifiGanConfig",
        "SpeechT5Processor",
    ],
    "models.splinter": [
        "SplinterConfig",
        "SplinterTokenizer",
    ],
    "models.squeezebert": [
        "SqueezeBertConfig",
        "SqueezeBertTokenizer",
    ],
    "models.stablelm": ["StableLmConfig"],
    "models.starcoder2": ["Starcoder2Config"],
    "models.superpoint": ["SuperPointConfig"],
    "models.swiftformer": ["SwiftFormerConfig"],
    "models.swin": ["SwinConfig"],
    "models.swin2sr": ["Swin2SRConfig"],
    "models.swinv2": ["Swinv2Config"],
    "models.switch_transformers": ["SwitchTransformersConfig"],
    "models.t5": ["T5Config"],
    "models.table_transformer": ["TableTransformerConfig"],
    "models.tapas": [
        "TapasConfig",
        "TapasTokenizer",
    ],
    "models.time_series_transformer": ["TimeSeriesTransformerConfig"],
    "models.timesformer": ["TimesformerConfig"],
    "models.timm_backbone": ["TimmBackboneConfig"],
    "models.trocr": [
        "TrOCRConfig",
        "TrOCRProcessor",
    ],
    "models.tvp": [
        "TvpConfig",
        "TvpProcessor",
    ],
    "models.udop": [
        "UdopConfig",
        "UdopProcessor",
    ],
    "models.umt5": ["UMT5Config"],
    "models.unispeech": ["UniSpeechConfig"],
    "models.unispeech_sat": ["UniSpeechSatConfig"],
    "models.univnet": [
        "UnivNetConfig",
        "UnivNetFeatureExtractor",
    ],
    "models.upernet": ["UperNetConfig"],
    "models.video_llava": ["VideoLlavaConfig"],
    "models.videomae": ["VideoMAEConfig"],
    "models.vilt": [
        "ViltConfig",
        "ViltFeatureExtractor",
        "ViltImageProcessor",
        "ViltProcessor",
    ],
    "models.vipllava": ["VipLlavaConfig"],
    "models.vision_encoder_decoder": ["VisionEncoderDecoderConfig"],
    "models.vision_text_dual_encoder": [
        "VisionTextDualEncoderConfig",
        "VisionTextDualEncoderProcessor",
    ],
    "models.visual_bert": ["VisualBertConfig"],
    "models.vit": ["ViTConfig"],
    "models.vit_mae": ["ViTMAEConfig"],
    "models.vit_msn": ["ViTMSNConfig"],
    "models.vitdet": ["VitDetConfig"],
    "models.vitmatte": ["VitMatteConfig"],
    "models.vits": [
        "VitsConfig",
        "VitsTokenizer",
    ],
    "models.vivit": ["VivitConfig"],
    "models.wav2vec2": [
        "Wav2Vec2Config",
        "Wav2Vec2CTCTokenizer",
        "Wav2Vec2FeatureExtractor",
        "Wav2Vec2Processor",
        "Wav2Vec2Tokenizer",
    ],
    "models.wav2vec2_bert": [
        "Wav2Vec2BertConfig",
        "Wav2Vec2BertProcessor",
    ],
    "models.wav2vec2_conformer": ["Wav2Vec2ConformerConfig"],
    "models.wav2vec2_phoneme": ["Wav2Vec2PhonemeCTCTokenizer"],
    "models.wav2vec2_with_lm": ["Wav2Vec2ProcessorWithLM"],
    "models.wavlm": ["WavLMConfig"],
    "models.whisper": [
        "WhisperConfig",
        "WhisperFeatureExtractor",
        "WhisperProcessor",
        "WhisperTokenizer",
    ],
    "models.x_clip": [
        "XCLIPConfig",
        "XCLIPProcessor",
        "XCLIPTextConfig",
        "XCLIPVisionConfig",
    ],
    "models.xglm": ["XGLMConfig"],
    "models.xlm": ["XLMConfig", "XLMTokenizer"],
    "models.xlm_roberta": ["XLMRobertaConfig"],
    "models.xlm_roberta_xl": ["XLMRobertaXLConfig"],
    "models.xlnet": ["XLNetConfig"],
    "models.xmod": ["XmodConfig"],
    "models.yolos": ["YolosConfig"],
    "models.yoso": ["YosoConfig"],
    "models.zamba": ["ZambaConfig"],
    "models.zoedepth": ["ZoeDepthConfig"],
    "onnx": [],
    "pipelines": [
        "AudioClassificationPipeline",
        "AutomaticSpeechRecognitionPipeline",
        "CsvPipelineDataFormat",
        "DepthEstimationPipeline",
        "DocumentQuestionAnsweringPipeline",
        "FeatureExtractionPipeline",
        "FillMaskPipeline",
        "ImageClassificationPipeline",
        "ImageFeatureExtractionPipeline",
        "ImageSegmentationPipeline",
        "ImageTextToTextPipeline",
        "ImageToImagePipeline",
        "ImageToTextPipeline",
        "JsonPipelineDataFormat",
        "MaskGenerationPipeline",
        "NerPipeline",
        "ObjectDetectionPipeline",
        "PipedPipelineDataFormat",
        "Pipeline",
        "PipelineDataFormat",
        "QuestionAnsweringPipeline",
        "SummarizationPipeline",
        "TableQuestionAnsweringPipeline",
        "Text2TextGenerationPipeline",
        "TextClassificationPipeline",
        "TextGenerationPipeline",
        "TextToAudioPipeline",
        "TokenClassificationPipeline",
        "TranslationPipeline",
        "VideoClassificationPipeline",
        "VisualQuestionAnsweringPipeline",
        "ZeroShotAudioClassificationPipeline",
        "ZeroShotClassificationPipeline",
        "ZeroShotImageClassificationPipeline",
        "ZeroShotObjectDetectionPipeline",
        "pipeline",
    ],
    "processing_utils": ["ProcessorMixin"],
    "quantizers": [],
    "testing_utils": [],
    "tokenization_utils": ["PreTrainedTokenizer"],
    "tokenization_utils_base": [
        "AddedToken",
        "BatchEncoding",
        "CharSpan",
        "PreTrainedTokenizerBase",
        "SpecialTokensMixin",
        "TokenSpan",
    ],
    "trainer_callback": [
        "DefaultFlowCallback",
        "EarlyStoppingCallback",
        "PrinterCallback",
        "ProgressCallback",
        "TrainerCallback",
        "TrainerControl",
        "TrainerState",
    ],
    "trainer_utils": [
        "EvalPrediction",
        "IntervalStrategy",
        "SchedulerType",
        "enable_full_determinism",
        "set_seed",
    ],
    "training_args": ["TrainingArguments"],
    "training_args_seq2seq": ["Seq2SeqTrainingArguments"],
    "training_args_tf": ["TFTrainingArguments"],
    "utils": [
        "CONFIG_NAME",
        "MODEL_CARD_NAME",
        "PYTORCH_PRETRAINED_BERT_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "SPIECE_UNDERLINE",
        "TF2_WEIGHTS_NAME",
        "TF_WEIGHTS_NAME",
        "TRANSFORMERS_CACHE",
        "WEIGHTS_NAME",
        "TensorType",
        "add_end_docstrings",
        "add_start_docstrings",
        "is_apex_available",
        "is_av_available",
        "is_bitsandbytes_available",
        "is_datasets_available",
        "is_faiss_available",
        "is_flax_available",
        "is_keras_nlp_available",
        "is_phonemizer_available",
        "is_psutil_available",
        "is_py3nvml_available",
        "is_pyctcdecode_available",
        "is_sacremoses_available",
        "is_safetensors_available",
        "is_scipy_available",
        "is_sentencepiece_available",
        "is_sklearn_available",
        "is_speech_available",
        "is_tensorflow_text_available",
        "is_tf_available",
        "is_timm_available",
        "is_tokenizers_available",
        "is_torch_available",
        "is_torch_mlu_available",
        "is_torch_musa_available",
        "is_torch_neuroncore_available",
        "is_torch_npu_available",
        "is_torch_tpu_available",
        "is_torchvision_available",
        "is_torch_xla_available",
        "is_torch_xpu_available",
        "is_vision_available",
        "logging",
    ],
    "utils.quantization_config": [
        "AqlmConfig",
        "AwqConfig",
        "BitNetConfig",
        "BitsAndBytesConfig",
        "CompressedTensorsConfig",
        "EetqConfig",
        "FbgemmFp8Config",
        "GPTQConfig",
        "HqqConfig",
        "QuantoConfig",
        "TorchAoConfig",
    ],
}

# sentencepiece-backed objects
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_sentencepiece_objects

    _import_structure["utils.dummy_sentencepiece_objects"] = [
        name for name in dir(dummy_sentencepiece_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.albert"].append("AlbertTokenizer")
    _import_structure["models.barthez"].append("BarthezTokenizer")
    _import_structure["models.bartpho"].append("BartphoTokenizer")
    _import_structure["models.bert_generation"].append("BertGenerationTokenizer")
    _import_structure["models.big_bird"].append("BigBirdTokenizer")
    _import_structure["models.camembert"].append("CamembertTokenizer")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizer")
    _import_structure["models.cpm"].append("CpmTokenizer")
    _import_structure["models.deberta_v2"].append("DebertaV2Tokenizer")
    _import_structure["models.deprecated.ernie_m"].append("ErnieMTokenizer")
    _import_structure["models.deprecated.xlm_prophetnet"].append("XLMProphetNetTokenizer")
    _import_structure["models.fnet"].append("FNetTokenizer")
    _import_structure["models.gemma"].append("GemmaTokenizer")
    _import_structure["models.gpt_sw3"].append("GPTSw3Tokenizer")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizer")
    _import_structure["models.llama"].append("LlamaTokenizer")
    _import_structure["models.m2m_100"].append("M2M100Tokenizer")
    _import_structure["models.marian"].append("MarianTokenizer")
    _import_structure["models.mbart"].append("MBartTokenizer")
    _import_structure["models.mbart50"].append("MBart50Tokenizer")
    _import_structure["models.mluke"].append("MLukeTokenizer")
    _import_structure["models.mt5"].append("MT5Tokenizer")
    _import_structure["models.nllb"].append("NllbTokenizer")
    _import_structure["models.pegasus"].append("PegasusTokenizer")
    _import_structure["models.plbart"].append("PLBartTokenizer")
    _import_structure["models.reformer"].append("ReformerTokenizer")
    _import_structure["models.rembert"].append("RemBertTokenizer")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizer")
    _import_structure["models.siglip"].append("SiglipTokenizer")
    _import_structure["models.speech_to_text"].append("Speech2TextTokenizer")
    _import_structure["models.speecht5"].append("SpeechT5Tokenizer")
    _import_structure["models.t5"].append("T5Tokenizer")
    _import_structure["models.udop"].append("UdopTokenizer")
    _import_structure["models.xglm"].append("XGLMTokenizer")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizer")
    _import_structure["models.xlnet"].append("XLNetTokenizer")

# tokenizers-backed objects
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_tokenizers_objects

    _import_structure["utils.dummy_tokenizers_objects"] = [
        name for name in dir(dummy_tokenizers_objects) if not name.startswith("_")
    ]
else:
    # Fast tokenizers structure
    _import_structure["models.albert"].append("AlbertTokenizerFast")
    _import_structure["models.bart"].append("BartTokenizerFast")
    _import_structure["models.barthez"].append("BarthezTokenizerFast")
    _import_structure["models.bert"].append("BertTokenizerFast")
    _import_structure["models.big_bird"].append("BigBirdTokenizerFast")
    _import_structure["models.blenderbot"].append("BlenderbotTokenizerFast")
    _import_structure["models.blenderbot_small"].append("BlenderbotSmallTokenizerFast")
    _import_structure["models.bloom"].append("BloomTokenizerFast")
    _import_structure["models.camembert"].append("CamembertTokenizerFast")
    _import_structure["models.clip"].append("CLIPTokenizerFast")
    _import_structure["models.code_llama"].append("CodeLlamaTokenizerFast")
    _import_structure["models.codegen"].append("CodeGenTokenizerFast")
    _import_structure["models.cohere"].append("CohereTokenizerFast")
    _import_structure["models.convbert"].append("ConvBertTokenizerFast")
    _import_structure["models.cpm"].append("CpmTokenizerFast")
    _import_structure["models.deberta"].append("DebertaTokenizerFast")
    _import_structure["models.deberta_v2"].append("DebertaV2TokenizerFast")
    _import_structure["models.deprecated.realm"].append("RealmTokenizerFast")
    _import_structure["models.deprecated.retribert"].append("RetriBertTokenizerFast")
    _import_structure["models.distilbert"].append("DistilBertTokenizerFast")
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoderTokenizerFast",
            "DPRQuestionEncoderTokenizerFast",
            "DPRReaderTokenizerFast",
        ]
    )
    _import_structure["models.electra"].append("ElectraTokenizerFast")
    _import_structure["models.fnet"].append("FNetTokenizerFast")
    _import_structure["models.funnel"].append("FunnelTokenizerFast")
    _import_structure["models.gemma"].append("GemmaTokenizerFast")
    _import_structure["models.gpt2"].append("GPT2TokenizerFast")
    _import_structure["models.gpt_neox"].append("GPTNeoXTokenizerFast")
    _import_structure["models.gpt_neox_japanese"].append("GPTNeoXJapaneseTokenizer")
    _import_structure["models.herbert"].append("HerbertTokenizerFast")
    _import_structure["models.layoutlm"].append("LayoutLMTokenizerFast")
    _import_structure["models.layoutlmv2"].append("LayoutLMv2TokenizerFast")
    _import_structure["models.layoutlmv3"].append("LayoutLMv3TokenizerFast")
    _import_structure["models.layoutxlm"].append("LayoutXLMTokenizerFast")
    _import_structure["models.led"].append("LEDTokenizerFast")
    _import_structure["models.llama"].append("LlamaTokenizerFast")
    _import_structure["models.longformer"].append("LongformerTokenizerFast")
    _import_structure["models.lxmert"].append("LxmertTokenizerFast")
    _import_structure["models.markuplm"].append("MarkupLMTokenizerFast")
    _import_structure["models.mbart"].append("MBartTokenizerFast")
    _import_structure["models.mbart50"].append("MBart50TokenizerFast")
    _import_structure["models.mobilebert"].append("MobileBertTokenizerFast")
    _import_structure["models.mpnet"].append("MPNetTokenizerFast")
    _import_structure["models.mt5"].append("MT5TokenizerFast")
    _import_structure["models.mvp"].append("MvpTokenizerFast")
    _import_structure["models.nllb"].append("NllbTokenizerFast")
    _import_structure["models.nougat"].append("NougatTokenizerFast")
    _import_structure["models.openai"].append("OpenAIGPTTokenizerFast")
    _import_structure["models.pegasus"].append("PegasusTokenizerFast")
    _import_structure["models.qwen2"].append("Qwen2TokenizerFast")
    _import_structure["models.reformer"].append("ReformerTokenizerFast")
    _import_structure["models.rembert"].append("RemBertTokenizerFast")
    _import_structure["models.roberta"].append("RobertaTokenizerFast")
    _import_structure["models.roformer"].append("RoFormerTokenizerFast")
    _import_structure["models.seamless_m4t"].append("SeamlessM4TTokenizerFast")
    _import_structure["models.splinter"].append("SplinterTokenizerFast")
    _import_structure["models.squeezebert"].append("SqueezeBertTokenizerFast")
    _import_structure["models.t5"].append("T5TokenizerFast")
    _import_structure["models.udop"].append("UdopTokenizerFast")
    _import_structure["models.whisper"].append("WhisperTokenizerFast")
    _import_structure["models.xglm"].append("XGLMTokenizerFast")
    _import_structure["models.xlm_roberta"].append("XLMRobertaTokenizerFast")
    _import_structure["models.xlnet"].append("XLNetTokenizerFast")
    _import_structure["tokenization_utils_fast"] = ["PreTrainedTokenizerFast"]


try:
    if not (is_sentencepiece_available() and is_tokenizers_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_sentencepiece_and_tokenizers_objects

    _import_structure["utils.dummy_sentencepiece_and_tokenizers_objects"] = [
        name for name in dir(dummy_sentencepiece_and_tokenizers_objects) if not name.startswith("_")
    ]
else:
    _import_structure["convert_slow_tokenizer"] = [
        "SLOW_TO_FAST_CONVERTERS",
        "convert_slow_tokenizer",
    ]

# Tensorflow-text-specific objects
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_tensorflow_text_objects

    _import_structure["utils.dummy_tensorflow_text_objects"] = [
        name for name in dir(dummy_tensorflow_text_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.bert"].append("TFBertTokenizer")

# keras-nlp-specific objects
try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_keras_nlp_objects

    _import_structure["utils.dummy_keras_nlp_objects"] = [
        name for name in dir(dummy_keras_nlp_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.gpt2"].append("TFGPT2Tokenizer")

# Vision-specific objects
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_vision_objects

    _import_structure["utils.dummy_vision_objects"] = [
        name for name in dir(dummy_vision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_base"] = ["ImageProcessingMixin"]
    _import_structure["image_processing_utils"] = ["BaseImageProcessor"]
    _import_structure["image_utils"] = ["ImageFeatureExtractionMixin"]
    _import_structure["models.beit"].extend(["BeitFeatureExtractor", "BeitImageProcessor"])
    _import_structure["models.bit"].extend(["BitImageProcessor"])
    _import_structure["models.blip"].extend(["BlipImageProcessor"])
    _import_structure["models.bridgetower"].append("BridgeTowerImageProcessor")
    _import_structure["models.chameleon"].append("ChameleonImageProcessor")
    _import_structure["models.chinese_clip"].extend(["ChineseCLIPFeatureExtractor", "ChineseCLIPImageProcessor"])
    _import_structure["models.clip"].extend(["CLIPFeatureExtractor", "CLIPImageProcessor"])
    _import_structure["models.conditional_detr"].extend(
        ["ConditionalDetrFeatureExtractor", "ConditionalDetrImageProcessor"]
    )
    _import_structure["models.convnext"].extend(["ConvNextFeatureExtractor", "ConvNextImageProcessor"])
    _import_structure["models.deformable_detr"].extend(
        ["DeformableDetrFeatureExtractor", "DeformableDetrImageProcessor"]
    )
    _import_structure["models.deit"].extend(["DeiTFeatureExtractor", "DeiTImageProcessor"])
    _import_structure["models.deprecated.deta"].append("DetaImageProcessor")
    _import_structure["models.deprecated.efficientformer"].append("EfficientFormerImageProcessor")
    _import_structure["models.deprecated.tvlt"].append("TvltImageProcessor")
    _import_structure["models.deprecated.vit_hybrid"].extend(["ViTHybridImageProcessor"])
    _import_structure["models.detr"].extend(["DetrFeatureExtractor", "DetrImageProcessor", "DetrImageProcessorFast"])
    _import_structure["models.donut"].extend(["DonutFeatureExtractor", "DonutImageProcessor"])
    _import_structure["models.dpt"].extend(["DPTFeatureExtractor", "DPTImageProcessor"])
    _import_structure["models.efficientnet"].append("EfficientNetImageProcessor")
    _import_structure["models.flava"].extend(["FlavaFeatureExtractor", "FlavaImageProcessor", "FlavaProcessor"])
    _import_structure["models.fuyu"].extend(["FuyuImageProcessor", "FuyuProcessor"])
    _import_structure["models.glpn"].extend(["GLPNFeatureExtractor", "GLPNImageProcessor"])
    _import_structure["models.grounding_dino"].extend(["GroundingDinoImageProcessor"])
    _import_structure["models.idefics"].extend(["IdeficsImageProcessor"])
    _import_structure["models.idefics2"].extend(["Idefics2ImageProcessor"])
    _import_structure["models.idefics3"].extend(["Idefics3ImageProcessor"])
    _import_structure["models.imagegpt"].extend(["ImageGPTFeatureExtractor", "ImageGPTImageProcessor"])
    _import_structure["models.instructblipvideo"].extend(["InstructBlipVideoImageProcessor"])
    _import_structure["models.layoutlmv2"].extend(["LayoutLMv2FeatureExtractor", "LayoutLMv2ImageProcessor"])
    _import_structure["models.layoutlmv3"].extend(["LayoutLMv3FeatureExtractor", "LayoutLMv3ImageProcessor"])
    _import_structure["models.levit"].extend(["LevitFeatureExtractor", "LevitImageProcessor"])
    _import_structure["models.llava_next"].append("LlavaNextImageProcessor")
    _import_structure["models.llava_next_video"].append("LlavaNextVideoImageProcessor")
    _import_structure["models.llava_onevision"].extend(
        ["LlavaOnevisionImageProcessor", "LlavaOnevisionVideoProcessor"]
    )
    _import_structure["models.mask2former"].append("Mask2FormerImageProcessor")
    _import_structure["models.maskformer"].extend(["MaskFormerFeatureExtractor", "MaskFormerImageProcessor"])
    _import_structure["models.mllama"].extend(["MllamaImageProcessor"])
    _import_structure["models.mobilenet_v1"].extend(["MobileNetV1FeatureExtractor", "MobileNetV1ImageProcessor"])
    _import_structure["models.mobilenet_v2"].extend(["MobileNetV2FeatureExtractor", "MobileNetV2ImageProcessor"])
    _import_structure["models.mobilevit"].extend(["MobileViTFeatureExtractor", "MobileViTImageProcessor"])
    _import_structure["models.nougat"].append("NougatImageProcessor")
    _import_structure["models.oneformer"].extend(["OneFormerImageProcessor"])
    _import_structure["models.owlv2"].append("Owlv2ImageProcessor")
    _import_structure["models.owlvit"].extend(["OwlViTFeatureExtractor", "OwlViTImageProcessor"])
    _import_structure["models.perceiver"].extend(["PerceiverFeatureExtractor", "PerceiverImageProcessor"])
    _import_structure["models.pix2struct"].extend(["Pix2StructImageProcessor"])
    _import_structure["models.pixtral"].append("PixtralImageProcessor")
    _import_structure["models.poolformer"].extend(["PoolFormerFeatureExtractor", "PoolFormerImageProcessor"])
    _import_structure["models.pvt"].extend(["PvtImageProcessor"])
    _import_structure["models.qwen2_vl"].extend(["Qwen2VLImageProcessor"])
    _import_structure["models.rt_detr"].extend(["RTDetrImageProcessor", "RTDetrImageProcessorFast"])
    _import_structure["models.sam"].extend(["SamImageProcessor"])
    _import_structure["models.segformer"].extend(["SegformerFeatureExtractor", "SegformerImageProcessor"])
    _import_structure["models.seggpt"].extend(["SegGptImageProcessor"])
    _import_structure["models.siglip"].append("SiglipImageProcessor")
    _import_structure["models.superpoint"].extend(["SuperPointImageProcessor"])
    _import_structure["models.swin2sr"].append("Swin2SRImageProcessor")
    _import_structure["models.tvp"].append("TvpImageProcessor")
    _import_structure["models.video_llava"].append("VideoLlavaImageProcessor")
    _import_structure["models.videomae"].extend(["VideoMAEFeatureExtractor", "VideoMAEImageProcessor"])
    _import_structure["models.vilt"].extend(["ViltFeatureExtractor", "ViltImageProcessor", "ViltProcessor"])
    _import_structure["models.vit"].extend(["ViTFeatureExtractor", "ViTImageProcessor"])
    _import_structure["models.vitmatte"].append("VitMatteImageProcessor")
    _import_structure["models.vivit"].append("VivitImageProcessor")
    _import_structure["models.yolos"].extend(["YolosFeatureExtractor", "YolosImageProcessor"])
    _import_structure["models.zoedepth"].append("ZoeDepthImageProcessor")

try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_torchvision_objects

    _import_structure["utils.dummy_torchvision_objects"] = [
        name for name in dir(dummy_torchvision_objects) if not name.startswith("_")
    ]
else:
    _import_structure["image_processing_utils_fast"] = ["BaseImageProcessorFast"]
    _import_structure["models.vit"].append("ViTImageProcessorFast")

# PyTorch-backed objects
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["activations"] = []
    _import_structure["benchmark.benchmark"] = ["PyTorchBenchmark"]
    _import_structure["benchmark.benchmark_args"] = ["PyTorchBenchmarkArguments"]
    _import_structure["cache_utils"] = [
        "Cache",
        "CacheConfig",
        "DynamicCache",
        "EncoderDecoderCache",
        "HQQQuantizedCache",
        "HybridCache",
        "MambaCache",
        "OffloadedCache",
        "OffloadedStaticCache",
        "QuantizedCache",
        "QuantizedCacheConfig",
        "QuantoQuantizedCache",
        "SinkCache",
        "SlidingWindowCache",
        "StaticCache",
    ]
    _import_structure["data.datasets"] = [
        "GlueDataset",
        "GlueDataTrainingArguments",
        "LineByLineTextDataset",
        "LineByLineWithRefDataset",
        "LineByLineWithSOPTextDataset",
        "SquadDataset",
        "SquadDataTrainingArguments",
        "TextDataset",
        "TextDatasetForNextSentencePrediction",
    ]
    _import_structure["generation"].extend(
        [
            "AlternatingCodebooksLogitsProcessor",
            "BayesianDetectorConfig",
            "BayesianDetectorModel",
            "BeamScorer",
            "BeamSearchScorer",
            "ClassifierFreeGuidanceLogitsProcessor",
            "ConstrainedBeamSearchScorer",
            "Constraint",
            "ConstraintListState",
            "DisjunctiveConstraint",
            "EncoderNoRepeatNGramLogitsProcessor",
            "EncoderRepetitionPenaltyLogitsProcessor",
            "EosTokenCriteria",
            "EpsilonLogitsWarper",
            "EtaLogitsWarper",
            "ExponentialDecayLengthPenalty",
            "ForcedBOSTokenLogitsProcessor",
            "ForcedEOSTokenLogitsProcessor",
            "GenerationMixin",
            "HammingDiversityLogitsProcessor",
            "InfNanRemoveLogitsProcessor",
            "LogitNormalization",
            "LogitsProcessor",
            "LogitsProcessorList",
            "LogitsWarper",
            "MaxLengthCriteria",
            "MaxTimeCriteria",
            "MinLengthLogitsProcessor",
            "MinNewTokensLengthLogitsProcessor",
            "MinPLogitsWarper",
            "NoBadWordsLogitsProcessor",
            "NoRepeatNGramLogitsProcessor",
            "PhrasalConstraint",
            "PrefixConstrainedLogitsProcessor",
            "RepetitionPenaltyLogitsProcessor",
            "SequenceBiasLogitsProcessor",
            "StoppingCriteria",
            "StoppingCriteriaList",
            "StopStringCriteria",
            "SuppressTokensAtBeginLogitsProcessor",
            "SuppressTokensLogitsProcessor",
            "SynthIDTextWatermarkDetector",
            "SynthIDTextWatermarkingConfig",
            "SynthIDTextWatermarkLogitsProcessor",
            "TemperatureLogitsWarper",
            "TopKLogitsWarper",
            "TopPLogitsWarper",
            "TypicalLogitsWarper",
            "UnbatchedClassifierFreeGuidanceLogitsProcessor",
            "WatermarkDetector",
            "WatermarkLogitsProcessor",
            "WhisperTimeStampLogitsProcessor",
        ]
    )

    # PyTorch domain libraries integration
    _import_structure["integrations.executorch"] = [
        "TorchExportableModuleWithStaticCache",
        "convert_and_export_with_cache",
    ]

    _import_structure["modeling_flash_attention_utils"] = []
    _import_structure["modeling_outputs"] = []
    _import_structure["modeling_rope_utils"] = ["ROPE_INIT_FUNCTIONS"]
    _import_structure["modeling_utils"] = ["PreTrainedModel"]

    # PyTorch models structure

    _import_structure["models.albert"].extend(
        [
            "AlbertForMaskedLM",
            "AlbertForMultipleChoice",
            "AlbertForPreTraining",
            "AlbertForQuestionAnswering",
            "AlbertForSequenceClassification",
            "AlbertForTokenClassification",
            "AlbertModel",
            "AlbertPreTrainedModel",
            "load_tf_weights_in_albert",
        ]
    )

    _import_structure["models.align"].extend(
        [
            "AlignModel",
            "AlignPreTrainedModel",
            "AlignTextModel",
            "AlignVisionModel",
        ]
    )
    _import_structure["models.altclip"].extend(
        [
            "AltCLIPModel",
            "AltCLIPPreTrainedModel",
            "AltCLIPTextModel",
            "AltCLIPVisionModel",
        ]
    )
    _import_structure["models.audio_spectrogram_transformer"].extend(
        [
            "ASTForAudioClassification",
            "ASTModel",
            "ASTPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING",
            "MODEL_FOR_AUDIO_XVECTOR_MAPPING",
            "MODEL_FOR_BACKBONE_MAPPING",
            "MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_CAUSAL_LM_MAPPING",
            "MODEL_FOR_CTC_MAPPING",
            "MODEL_FOR_DEPTH_ESTIMATION_MAPPING",
            "MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_IMAGE_MAPPING",
            "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING",
            "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING",
            "MODEL_FOR_IMAGE_TO_IMAGE_MAPPING",
            "MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING",
            "MODEL_FOR_KEYPOINT_DETECTION_MAPPING",
            "MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "MODEL_FOR_MASKED_LM_MAPPING",
            "MODEL_FOR_MASK_GENERATION_MAPPING",
            "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "MODEL_FOR_OBJECT_DETECTION_MAPPING",
            "MODEL_FOR_PRETRAINING_MAPPING",
            "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_TEXT_ENCODING_MAPPING",
            "MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING",
            "MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING",
            "MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING",
            "MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING",
            "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING",
            "MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING",
            "MODEL_FOR_VISION_2_SEQ_MAPPING",
            "MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING",
            "MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING",
            "MODEL_MAPPING",
            "MODEL_WITH_LM_HEAD_MAPPING",
            "AutoBackbone",
            "AutoModel",
            "AutoModelForAudioClassification",
            "AutoModelForAudioFrameClassification",
            "AutoModelForAudioXVector",
            "AutoModelForCausalLM",
            "AutoModelForCTC",
            "AutoModelForDepthEstimation",
            "AutoModelForDocumentQuestionAnswering",
            "AutoModelForImageClassification",
            "AutoModelForImageSegmentation",
            "AutoModelForImageTextToText",
            "AutoModelForImageToImage",
            "AutoModelForInstanceSegmentation",
            "AutoModelForKeypointDetection",
            "AutoModelForMaskedImageModeling",
            "AutoModelForMaskedLM",
            "AutoModelForMaskGeneration",
            "AutoModelForMultipleChoice",
            "AutoModelForNextSentencePrediction",
            "AutoModelForObjectDetection",
            "AutoModelForPreTraining",
            "AutoModelForQuestionAnswering",
            "AutoModelForSemanticSegmentation",
            "AutoModelForSeq2SeqLM",
            "AutoModelForSequenceClassification",
            "AutoModelForSpeechSeq2Seq",
            "AutoModelForTableQuestionAnswering",
            "AutoModelForTextEncoding",
            "AutoModelForTextToSpectrogram",
            "AutoModelForTextToWaveform",
            "AutoModelForTokenClassification",
            "AutoModelForUniversalSegmentation",
            "AutoModelForVideoClassification",
            "AutoModelForVision2Seq",
            "AutoModelForVisualQuestionAnswering",
            "AutoModelForZeroShotImageClassification",
            "AutoModelForZeroShotObjectDetection",
            "AutoModelWithLMHead",
        ]
    )
    _import_structure["models.autoformer"].extend(
        [
            "AutoformerForPrediction",
            "AutoformerModel",
            "AutoformerPreTrainedModel",
        ]
    )
    _import_structure["models.bark"].extend(
        [
            "BarkCausalModel",
            "BarkCoarseModel",
            "BarkFineModel",
            "BarkModel",
            "BarkPreTrainedModel",
            "BarkSemanticModel",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "BartForCausalLM",
            "BartForConditionalGeneration",
            "BartForQuestionAnswering",
            "BartForSequenceClassification",
            "BartModel",
            "BartPretrainedModel",
            "BartPreTrainedModel",
            "PretrainedBartModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "BeitBackbone",
            "BeitForImageClassification",
            "BeitForMaskedImageModeling",
            "BeitForSemanticSegmentation",
            "BeitModel",
            "BeitPreTrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertForNextSentencePrediction",
            "BertForPreTraining",
            "BertForQuestionAnswering",
            "BertForSequenceClassification",
            "BertForTokenClassification",
            "BertLMHeadModel",
            "BertModel",
            "BertPreTrainedModel",
            "load_tf_weights_in_bert",
        ]
    )
    _import_structure["models.bert_generation"].extend(
        [
            "BertGenerationDecoder",
            "BertGenerationEncoder",
            "BertGenerationPreTrainedModel",
            "load_tf_weights_in_bert_generation",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "BigBirdForCausalLM",
            "BigBirdForMaskedLM",
            "BigBirdForMultipleChoice",
            "BigBirdForPreTraining",
            "BigBirdForQuestionAnswering",
            "BigBirdForSequenceClassification",
            "BigBirdForTokenClassification",
            "BigBirdModel",
            "BigBirdPreTrainedModel",
            "load_tf_weights_in_big_bird",
        ]
    )
    _import_structure["models.bigbird_pegasus"].extend(
        [
            "BigBirdPegasusForCausalLM",
            "BigBirdPegasusForConditionalGeneration",
            "BigBirdPegasusForQuestionAnswering",
            "BigBirdPegasusForSequenceClassification",
            "BigBirdPegasusModel",
            "BigBirdPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.biogpt"].extend(
        [
            "BioGptForCausalLM",
            "BioGptForSequenceClassification",
            "BioGptForTokenClassification",
            "BioGptModel",
            "BioGptPreTrainedModel",
        ]
    )
    _import_structure["models.bit"].extend(
        [
            "BitBackbone",
            "BitForImageClassification",
            "BitModel",
            "BitPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "BlenderbotForCausalLM",
            "BlenderbotForConditionalGeneration",
            "BlenderbotModel",
            "BlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "BlenderbotSmallForCausalLM",
            "BlenderbotSmallForConditionalGeneration",
            "BlenderbotSmallModel",
            "BlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
            "BlipForConditionalGeneration",
            "BlipForImageTextRetrieval",
            "BlipForQuestionAnswering",
            "BlipModel",
            "BlipPreTrainedModel",
            "BlipTextModel",
            "BlipVisionModel",
        ]
    )
    _import_structure["models.blip_2"].extend(
        [
            "Blip2ForConditionalGeneration",
            "Blip2ForImageTextRetrieval",
            "Blip2Model",
            "Blip2PreTrainedModel",
            "Blip2QFormerModel",
            "Blip2TextModelWithProjection",
            "Blip2VisionModel",
            "Blip2VisionModelWithProjection",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "BloomForCausalLM",
            "BloomForQuestionAnswering",
            "BloomForSequenceClassification",
            "BloomForTokenClassification",
            "BloomModel",
            "BloomPreTrainedModel",
        ]
    )
    _import_structure["models.bridgetower"].extend(
        [
            "BridgeTowerForContrastiveLearning",
            "BridgeTowerForImageAndTextRetrieval",
            "BridgeTowerForMaskedLM",
            "BridgeTowerModel",
            "BridgeTowerPreTrainedModel",
        ]
    )
    _import_structure["models.bros"].extend(
        [
            "BrosForTokenClassification",
            "BrosModel",
            "BrosPreTrainedModel",
            "BrosProcessor",
            "BrosSpadeEEForTokenClassification",
            "BrosSpadeELForTokenClassification",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "CamembertForCausalLM",
            "CamembertForMaskedLM",
            "CamembertForMultipleChoice",
            "CamembertForQuestionAnswering",
            "CamembertForSequenceClassification",
            "CamembertForTokenClassification",
            "CamembertModel",
            "CamembertPreTrainedModel",
        ]
    )
    _import_structure["models.canine"].extend(
        [
            "CanineForMultipleChoice",
            "CanineForQuestionAnswering",
            "CanineForSequenceClassification",
            "CanineForTokenClassification",
            "CanineModel",
            "CaninePreTrainedModel",
            "load_tf_weights_in_canine",
        ]
    )
    _import_structure["models.chameleon"].extend(
        [
            "ChameleonForConditionalGeneration",
            "ChameleonModel",
            "ChameleonPreTrainedModel",
            "ChameleonProcessor",
            "ChameleonVQVAE",
        ]
    )
    _import_structure["models.chinese_clip"].extend(
        [
            "ChineseCLIPModel",
            "ChineseCLIPPreTrainedModel",
            "ChineseCLIPTextModel",
            "ChineseCLIPVisionModel",
        ]
    )
    _import_structure["models.clap"].extend(
        [
            "ClapAudioModel",
            "ClapAudioModelWithProjection",
            "ClapFeatureExtractor",
            "ClapModel",
            "ClapPreTrainedModel",
            "ClapTextModel",
            "ClapTextModelWithProjection",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "CLIPForImageClassification",
            "CLIPModel",
            "CLIPPreTrainedModel",
            "CLIPTextModel",
            "CLIPTextModelWithProjection",
            "CLIPVisionModel",
            "CLIPVisionModelWithProjection",
        ]
    )
    _import_structure["models.clipseg"].extend(
        [
            "CLIPSegForImageSegmentation",
            "CLIPSegModel",
            "CLIPSegPreTrainedModel",
            "CLIPSegTextModel",
            "CLIPSegVisionModel",
        ]
    )
    _import_structure["models.clvp"].extend(
        [
            "ClvpDecoder",
            "ClvpEncoder",
            "ClvpForCausalLM",
            "ClvpModel",
            "ClvpModelForConditionalGeneration",
            "ClvpPreTrainedModel",
        ]
    )
    _import_structure["models.codegen"].extend(
        [
            "CodeGenForCausalLM",
            "CodeGenModel",
            "CodeGenPreTrainedModel",
        ]
    )
    _import_structure["models.cohere"].extend(["CohereForCausalLM", "CohereModel", "CoherePreTrainedModel"])
    _import_structure["models.conditional_detr"].extend(
        [
            "ConditionalDetrForObjectDetection",
            "ConditionalDetrForSegmentation",
            "ConditionalDetrModel",
            "ConditionalDetrPreTrainedModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "ConvBertForMaskedLM",
            "ConvBertForMultipleChoice",
            "ConvBertForQuestionAnswering",
            "ConvBertForSequenceClassification",
            "ConvBertForTokenClassification",
            "ConvBertModel",
            "ConvBertPreTrainedModel",
            "load_tf_weights_in_convbert",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "ConvNextBackbone",
            "ConvNextForImageClassification",
            "ConvNextModel",
            "ConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "ConvNextV2Backbone",
            "ConvNextV2ForImageClassification",
            "ConvNextV2Model",
            "ConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.cpmant"].extend(
        [
            "CpmAntForCausalLM",
            "CpmAntModel",
            "CpmAntPreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "CTRLForSequenceClassification",
            "CTRLLMHeadModel",
            "CTRLModel",
            "CTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "CvtForImageClassification",
            "CvtModel",
            "CvtPreTrainedModel",
        ]
    )
    _import_structure["models.dac"].extend(
        [
            "DacModel",
            "DacPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "Data2VecAudioForAudioFrameClassification",
            "Data2VecAudioForCTC",
            "Data2VecAudioForSequenceClassification",
            "Data2VecAudioForXVector",
            "Data2VecAudioModel",
            "Data2VecAudioPreTrainedModel",
            "Data2VecTextForCausalLM",
            "Data2VecTextForMaskedLM",
            "Data2VecTextForMultipleChoice",
            "Data2VecTextForQuestionAnswering",
            "Data2VecTextForSequenceClassification",
            "Data2VecTextForTokenClassification",
            "Data2VecTextModel",
            "Data2VecTextPreTrainedModel",
            "Data2VecVisionForImageClassification",
            "Data2VecVisionForSemanticSegmentation",
            "Data2VecVisionModel",
            "Data2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.dbrx"].extend(
        [
            "DbrxForCausalLM",
            "DbrxModel",
            "DbrxPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "DebertaForMaskedLM",
            "DebertaForQuestionAnswering",
            "DebertaForSequenceClassification",
            "DebertaForTokenClassification",
            "DebertaModel",
            "DebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "DebertaV2ForMaskedLM",
            "DebertaV2ForMultipleChoice",
            "DebertaV2ForQuestionAnswering",
            "DebertaV2ForSequenceClassification",
            "DebertaV2ForTokenClassification",
            "DebertaV2Model",
            "DebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.decision_transformer"].extend(
        [
            "DecisionTransformerGPT2Model",
            "DecisionTransformerGPT2PreTrainedModel",
            "DecisionTransformerModel",
            "DecisionTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deformable_detr"].extend(
        [
            "DeformableDetrForObjectDetection",
            "DeformableDetrModel",
            "DeformableDetrPreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "DeiTForImageClassification",
            "DeiTForImageClassificationWithTeacher",
            "DeiTForMaskedImageModeling",
            "DeiTModel",
            "DeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.deta"].extend(
        [
            "DetaForObjectDetection",
            "DetaModel",
            "DetaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "EfficientFormerForImageClassification",
            "EfficientFormerForImageClassificationWithTeacher",
            "EfficientFormerModel",
            "EfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.ernie_m"].extend(
        [
            "ErnieMForInformationExtraction",
            "ErnieMForMultipleChoice",
            "ErnieMForQuestionAnswering",
            "ErnieMForSequenceClassification",
            "ErnieMForTokenClassification",
            "ErnieMModel",
            "ErnieMPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.gptsan_japanese"].extend(
        [
            "GPTSanJapaneseForConditionalGeneration",
            "GPTSanJapaneseModel",
            "GPTSanJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.graphormer"].extend(
        [
            "GraphormerForGraphClassification",
            "GraphormerModel",
            "GraphormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.jukebox"].extend(
        [
            "JukeboxModel",
            "JukeboxPreTrainedModel",
            "JukeboxPrior",
            "JukeboxVQVAE",
        ]
    )
    _import_structure["models.deprecated.mctct"].extend(
        [
            "MCTCTForCTC",
            "MCTCTModel",
            "MCTCTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mega"].extend(
        [
            "MegaForCausalLM",
            "MegaForMaskedLM",
            "MegaForMultipleChoice",
            "MegaForQuestionAnswering",
            "MegaForSequenceClassification",
            "MegaForTokenClassification",
            "MegaModel",
            "MegaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.mmbt"].extend(["MMBTForClassification", "MMBTModel", "ModalEmbeddings"])
    _import_structure["models.deprecated.nat"].extend(
        [
            "NatBackbone",
            "NatForImageClassification",
            "NatModel",
            "NatPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.nezha"].extend(
        [
            "NezhaForMaskedLM",
            "NezhaForMultipleChoice",
            "NezhaForNextSentencePrediction",
            "NezhaForPreTraining",
            "NezhaForQuestionAnswering",
            "NezhaForSequenceClassification",
            "NezhaForTokenClassification",
            "NezhaModel",
            "NezhaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.open_llama"].extend(
        [
            "OpenLlamaForCausalLM",
            "OpenLlamaForSequenceClassification",
            "OpenLlamaModel",
            "OpenLlamaPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.qdqbert"].extend(
        [
            "QDQBertForMaskedLM",
            "QDQBertForMultipleChoice",
            "QDQBertForNextSentencePrediction",
            "QDQBertForQuestionAnswering",
            "QDQBertForSequenceClassification",
            "QDQBertForTokenClassification",
            "QDQBertLMHeadModel",
            "QDQBertModel",
            "QDQBertPreTrainedModel",
            "load_tf_weights_in_qdqbert",
        ]
    )
    _import_structure["models.deprecated.realm"].extend(
        [
            "RealmEmbedder",
            "RealmForOpenQA",
            "RealmKnowledgeAugEncoder",
            "RealmPreTrainedModel",
            "RealmReader",
            "RealmRetriever",
            "RealmScorer",
            "load_tf_weights_in_realm",
        ]
    )
    _import_structure["models.deprecated.retribert"].extend(
        [
            "RetriBertModel",
            "RetriBertPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.speech_to_text_2"].extend(
        ["Speech2Text2ForCausalLM", "Speech2Text2PreTrainedModel"]
    )
    _import_structure["models.deprecated.trajectory_transformer"].extend(
        [
            "TrajectoryTransformerModel",
            "TrajectoryTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "AdaptiveEmbedding",
            "TransfoXLForSequenceClassification",
            "TransfoXLLMHeadModel",
            "TransfoXLModel",
            "TransfoXLPreTrainedModel",
            "load_tf_weights_in_transfo_xl",
        ]
    )
    _import_structure["models.deprecated.tvlt"].extend(
        [
            "TvltForAudioVisualClassification",
            "TvltForPreTraining",
            "TvltModel",
            "TvltPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.van"].extend(
        [
            "VanForImageClassification",
            "VanModel",
            "VanPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.vit_hybrid"].extend(
        [
            "ViTHybridForImageClassification",
            "ViTHybridModel",
            "ViTHybridPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.xlm_prophetnet"].extend(
        [
            "XLMProphetNetDecoder",
            "XLMProphetNetEncoder",
            "XLMProphetNetForCausalLM",
            "XLMProphetNetForConditionalGeneration",
            "XLMProphetNetModel",
            "XLMProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.depth_anything"].extend(
        [
            "DepthAnythingForDepthEstimation",
            "DepthAnythingPreTrainedModel",
        ]
    )
    _import_structure["models.detr"].extend(
        [
            "DetrForObjectDetection",
            "DetrForSegmentation",
            "DetrModel",
            "DetrPreTrainedModel",
        ]
    )
    _import_structure["models.dinat"].extend(
        [
            "DinatBackbone",
            "DinatForImageClassification",
            "DinatModel",
            "DinatPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "Dinov2Backbone",
            "Dinov2ForImageClassification",
            "Dinov2Model",
            "Dinov2PreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "DistilBertForMaskedLM",
            "DistilBertForMultipleChoice",
            "DistilBertForQuestionAnswering",
            "DistilBertForSequenceClassification",
            "DistilBertForTokenClassification",
            "DistilBertModel",
            "DistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.donut"].extend(
        [
            "DonutSwinModel",
            "DonutSwinPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "DPRContextEncoder",
            "DPRPretrainedContextEncoder",
            "DPRPreTrainedModel",
            "DPRPretrainedQuestionEncoder",
            "DPRPretrainedReader",
            "DPRQuestionEncoder",
            "DPRReader",
        ]
    )
    _import_structure["models.dpt"].extend(
        [
            "DPTForDepthEstimation",
            "DPTForSemanticSegmentation",
            "DPTModel",
            "DPTPreTrainedModel",
        ]
    )
    _import_structure["models.efficientnet"].extend(
        [
            "EfficientNetForImageClassification",
            "EfficientNetModel",
            "EfficientNetPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "ElectraForCausalLM",
            "ElectraForMaskedLM",
            "ElectraForMultipleChoice",
            "ElectraForPreTraining",
            "ElectraForQuestionAnswering",
            "ElectraForSequenceClassification",
            "ElectraForTokenClassification",
            "ElectraModel",
            "ElectraPreTrainedModel",
            "load_tf_weights_in_electra",
        ]
    )
    _import_structure["models.encodec"].extend(
        [
            "EncodecModel",
            "EncodecPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("EncoderDecoderModel")
    _import_structure["models.ernie"].extend(
        [
            "ErnieForCausalLM",
            "ErnieForMaskedLM",
            "ErnieForMultipleChoice",
            "ErnieForNextSentencePrediction",
            "ErnieForPreTraining",
            "ErnieForQuestionAnswering",
            "ErnieForSequenceClassification",
            "ErnieForTokenClassification",
            "ErnieModel",
            "ErniePreTrainedModel",
        ]
    )
    _import_structure["models.esm"].extend(
        [
            "EsmFoldPreTrainedModel",
            "EsmForMaskedLM",
            "EsmForProteinFolding",
            "EsmForSequenceClassification",
            "EsmForTokenClassification",
            "EsmModel",
            "EsmPreTrainedModel",
        ]
    )
    _import_structure["models.falcon"].extend(
        [
            "FalconForCausalLM",
            "FalconForQuestionAnswering",
            "FalconForSequenceClassification",
            "FalconForTokenClassification",
            "FalconModel",
            "FalconPreTrainedModel",
        ]
    )
    _import_structure["models.falcon_mamba"].extend(
        [
            "FalconMambaForCausalLM",
            "FalconMambaModel",
            "FalconMambaPreTrainedModel",
        ]
    )
    _import_structure["models.fastspeech2_conformer"].extend(
        [
            "FastSpeech2ConformerHifiGan",
            "FastSpeech2ConformerModel",
            "FastSpeech2ConformerPreTrainedModel",
            "FastSpeech2ConformerWithHifiGan",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "FlaubertForMultipleChoice",
            "FlaubertForQuestionAnswering",
            "FlaubertForQuestionAnsweringSimple",
            "FlaubertForSequenceClassification",
            "FlaubertForTokenClassification",
            "FlaubertModel",
            "FlaubertPreTrainedModel",
            "FlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.flava"].extend(
        [
            "FlavaForPreTraining",
            "FlavaImageCodebook",
            "FlavaImageModel",
            "FlavaModel",
            "FlavaMultimodalModel",
            "FlavaPreTrainedModel",
            "FlavaTextModel",
        ]
    )
    _import_structure["models.fnet"].extend(
        [
            "FNetForMaskedLM",
            "FNetForMultipleChoice",
            "FNetForNextSentencePrediction",
            "FNetForPreTraining",
            "FNetForQuestionAnswering",
            "FNetForSequenceClassification",
            "FNetForTokenClassification",
            "FNetModel",
            "FNetPreTrainedModel",
        ]
    )
    _import_structure["models.focalnet"].extend(
        [
            "FocalNetBackbone",
            "FocalNetForImageClassification",
            "FocalNetForMaskedImageModeling",
            "FocalNetModel",
            "FocalNetPreTrainedModel",
        ]
    )
    _import_structure["models.fsmt"].extend(["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"])
    _import_structure["models.funnel"].extend(
        [
            "FunnelBaseModel",
            "FunnelForMaskedLM",
            "FunnelForMultipleChoice",
            "FunnelForPreTraining",
            "FunnelForQuestionAnswering",
            "FunnelForSequenceClassification",
            "FunnelForTokenClassification",
            "FunnelModel",
            "FunnelPreTrainedModel",
            "load_tf_weights_in_funnel",
        ]
    )
    _import_structure["models.fuyu"].extend(["FuyuForCausalLM", "FuyuPreTrainedModel"])
    _import_structure["models.gemma"].extend(
        [
            "GemmaForCausalLM",
            "GemmaForSequenceClassification",
            "GemmaForTokenClassification",
            "GemmaModel",
            "GemmaPreTrainedModel",
        ]
    )
    _import_structure["models.gemma2"].extend(
        [
            "Gemma2ForCausalLM",
            "Gemma2ForSequenceClassification",
            "Gemma2ForTokenClassification",
            "Gemma2Model",
            "Gemma2PreTrainedModel",
        ]
    )
    _import_structure["models.git"].extend(
        [
            "GitForCausalLM",
            "GitModel",
            "GitPreTrainedModel",
            "GitVisionModel",
        ]
    )
    _import_structure["models.glm"].extend(
        [
            "GlmForCausalLM",
            "GlmForSequenceClassification",
            "GlmForTokenClassification",
            "GlmModel",
            "GlmPreTrainedModel",
        ]
    )
    _import_structure["models.glpn"].extend(
        [
            "GLPNForDepthEstimation",
            "GLPNModel",
            "GLPNPreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "GPT2DoubleHeadsModel",
            "GPT2ForQuestionAnswering",
            "GPT2ForSequenceClassification",
            "GPT2ForTokenClassification",
            "GPT2LMHeadModel",
            "GPT2Model",
            "GPT2PreTrainedModel",
            "load_tf_weights_in_gpt2",
        ]
    )
    _import_structure["models.gpt_bigcode"].extend(
        [
            "GPTBigCodeForCausalLM",
            "GPTBigCodeForSequenceClassification",
            "GPTBigCodeForTokenClassification",
            "GPTBigCodeModel",
            "GPTBigCodePreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neo"].extend(
        [
            "GPTNeoForCausalLM",
            "GPTNeoForQuestionAnswering",
            "GPTNeoForSequenceClassification",
            "GPTNeoForTokenClassification",
            "GPTNeoModel",
            "GPTNeoPreTrainedModel",
            "load_tf_weights_in_gpt_neo",
        ]
    )
    _import_structure["models.gpt_neox"].extend(
        [
            "GPTNeoXForCausalLM",
            "GPTNeoXForQuestionAnswering",
            "GPTNeoXForSequenceClassification",
            "GPTNeoXForTokenClassification",
            "GPTNeoXModel",
            "GPTNeoXPreTrainedModel",
        ]
    )
    _import_structure["models.gpt_neox_japanese"].extend(
        [
            "GPTNeoXJapaneseForCausalLM",
            "GPTNeoXJapaneseModel",
            "GPTNeoXJapanesePreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "GPTJForCausalLM",
            "GPTJForQuestionAnswering",
            "GPTJForSequenceClassification",
            "GPTJModel",
            "GPTJPreTrainedModel",
        ]
    )
    _import_structure["models.granite"].extend(
        [
            "GraniteForCausalLM",
            "GraniteModel",
            "GranitePreTrainedModel",
        ]
    )
    _import_structure["models.granitemoe"].extend(
        [
            "GraniteMoeForCausalLM",
            "GraniteMoeModel",
            "GraniteMoePreTrainedModel",
        ]
    )
    _import_structure["models.grounding_dino"].extend(
        [
            "GroundingDinoForObjectDetection",
            "GroundingDinoModel",
            "GroundingDinoPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "GroupViTModel",
            "GroupViTPreTrainedModel",
            "GroupViTTextModel",
            "GroupViTVisionModel",
        ]
    )
    _import_structure["models.hiera"].extend(
        [
            "HieraBackbone",
            "HieraForImageClassification",
            "HieraForPreTraining",
            "HieraModel",
            "HieraPreTrainedModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "HubertForCTC",
            "HubertForSequenceClassification",
            "HubertModel",
            "HubertPreTrainedModel",
        ]
    )
    _import_structure["models.ibert"].extend(
        [
            "IBertForMaskedLM",
            "IBertForMultipleChoice",
            "IBertForQuestionAnswering",
            "IBertForSequenceClassification",
            "IBertForTokenClassification",
            "IBertModel",
            "IBertPreTrainedModel",
        ]
    )
    _import_structure["models.idefics"].extend(
        [
            "IdeficsForVisionText2Text",
            "IdeficsModel",
            "IdeficsPreTrainedModel",
            "IdeficsProcessor",
        ]
    )
    _import_structure["models.idefics2"].extend(
        [
            "Idefics2ForConditionalGeneration",
            "Idefics2Model",
            "Idefics2PreTrainedModel",
            "Idefics2Processor",
        ]
    )
    _import_structure["models.idefics3"].extend(
        [
            "Idefics3ForConditionalGeneration",
            "Idefics3Model",
            "Idefics3PreTrainedModel",
            "Idefics3Processor",
        ]
    )
    _import_structure["models.imagegpt"].extend(
        [
            "ImageGPTForCausalImageModeling",
            "ImageGPTForImageClassification",
            "ImageGPTModel",
            "ImageGPTPreTrainedModel",
            "load_tf_weights_in_imagegpt",
        ]
    )
    _import_structure["models.informer"].extend(
        [
            "InformerForPrediction",
            "InformerModel",
            "InformerPreTrainedModel",
        ]
    )
    _import_structure["models.instructblip"].extend(
        [
            "InstructBlipForConditionalGeneration",
            "InstructBlipPreTrainedModel",
            "InstructBlipQFormerModel",
            "InstructBlipVisionModel",
        ]
    )
    _import_structure["models.instructblipvideo"].extend(
        [
            "InstructBlipVideoForConditionalGeneration",
            "InstructBlipVideoPreTrainedModel",
            "InstructBlipVideoQFormerModel",
            "InstructBlipVideoVisionModel",
        ]
    )
    _import_structure["models.jamba"].extend(
        [
            "JambaForCausalLM",
            "JambaForSequenceClassification",
            "JambaModel",
            "JambaPreTrainedModel",
        ]
    )
    _import_structure["models.jetmoe"].extend(
        [
            "JetMoeForCausalLM",
            "JetMoeForSequenceClassification",
            "JetMoeModel",
            "JetMoePreTrainedModel",
        ]
    )
    _import_structure["models.kosmos2"].extend(
        [
            "Kosmos2ForConditionalGeneration",
            "Kosmos2Model",
            "Kosmos2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlm"].extend(
        [
            "LayoutLMForMaskedLM",
            "LayoutLMForQuestionAnswering",
            "LayoutLMForSequenceClassification",
            "LayoutLMForTokenClassification",
            "LayoutLMModel",
            "LayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv2"].extend(
        [
            "LayoutLMv2ForQuestionAnswering",
            "LayoutLMv2ForSequenceClassification",
            "LayoutLMv2ForTokenClassification",
            "LayoutLMv2Model",
            "LayoutLMv2PreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "LayoutLMv3ForQuestionAnswering",
            "LayoutLMv3ForSequenceClassification",
            "LayoutLMv3ForTokenClassification",
            "LayoutLMv3Model",
            "LayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(
        [
            "LEDForConditionalGeneration",
            "LEDForQuestionAnswering",
            "LEDForSequenceClassification",
            "LEDModel",
            "LEDPreTrainedModel",
        ]
    )
    _import_structure["models.levit"].extend(
        [
            "LevitForImageClassification",
            "LevitForImageClassificationWithTeacher",
            "LevitModel",
            "LevitPreTrainedModel",
        ]
    )
    _import_structure["models.lilt"].extend(
        [
            "LiltForQuestionAnswering",
            "LiltForSequenceClassification",
            "LiltForTokenClassification",
            "LiltModel",
            "LiltPreTrainedModel",
        ]
    )
    _import_structure["models.llama"].extend(
        [
            "LlamaForCausalLM",
            "LlamaForQuestionAnswering",
            "LlamaForSequenceClassification",
            "LlamaForTokenClassification",
            "LlamaModel",
            "LlamaPreTrainedModel",
        ]
    )
    _import_structure["models.llava"].extend(
        [
            "LlavaForConditionalGeneration",
            "LlavaPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next"].extend(
        [
            "LlavaNextForConditionalGeneration",
            "LlavaNextPreTrainedModel",
        ]
    )
    _import_structure["models.llava_next_video"].extend(
        [
            "LlavaNextVideoForConditionalGeneration",
            "LlavaNextVideoPreTrainedModel",
        ]
    )
    _import_structure["models.llava_onevision"].extend(
        [
            "LlavaOnevisionForConditionalGeneration",
            "LlavaOnevisionPreTrainedModel",
        ]
    )
    _import_structure["models.longformer"].extend(
        [
            "LongformerForMaskedLM",
            "LongformerForMultipleChoice",
            "LongformerForQuestionAnswering",
            "LongformerForSequenceClassification",
            "LongformerForTokenClassification",
            "LongformerModel",
            "LongformerPreTrainedModel",
        ]
    )
    _import_structure["models.longt5"].extend(
        [
            "LongT5EncoderModel",
            "LongT5ForConditionalGeneration",
            "LongT5Model",
            "LongT5PreTrainedModel",
        ]
    )
    _import_structure["models.luke"].extend(
        [
            "LukeForEntityClassification",
            "LukeForEntityPairClassification",
            "LukeForEntitySpanClassification",
            "LukeForMaskedLM",
            "LukeForMultipleChoice",
            "LukeForQuestionAnswering",
            "LukeForSequenceClassification",
            "LukeForTokenClassification",
            "LukeModel",
            "LukePreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "LxmertEncoder",
            "LxmertForPreTraining",
            "LxmertForQuestionAnswering",
            "LxmertModel",
            "LxmertPreTrainedModel",
            "LxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.m2m_100"].extend(
        [
            "M2M100ForConditionalGeneration",
            "M2M100Model",
            "M2M100PreTrainedModel",
        ]
    )
    _import_structure["models.mamba"].extend(
        [
            "MambaForCausalLM",
            "MambaModel",
            "MambaPreTrainedModel",
        ]
    )
    _import_structure["models.mamba2"].extend(
        [
            "Mamba2ForCausalLM",
            "Mamba2Model",
            "Mamba2PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(
        ["MarianForCausalLM", "MarianModel", "MarianMTModel", "MarianPreTrainedModel"]
    )
    _import_structure["models.markuplm"].extend(
        [
            "MarkupLMForQuestionAnswering",
            "MarkupLMForSequenceClassification",
            "MarkupLMForTokenClassification",
            "MarkupLMModel",
            "MarkupLMPreTrainedModel",
        ]
    )
    _import_structure["models.mask2former"].extend(
        [
            "Mask2FormerForUniversalSegmentation",
            "Mask2FormerModel",
            "Mask2FormerPreTrainedModel",
        ]
    )
    _import_structure["models.maskformer"].extend(
        [
            "MaskFormerForInstanceSegmentation",
            "MaskFormerModel",
            "MaskFormerPreTrainedModel",
            "MaskFormerSwinBackbone",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "MBartForCausalLM",
            "MBartForConditionalGeneration",
            "MBartForQuestionAnswering",
            "MBartForSequenceClassification",
            "MBartModel",
            "MBartPreTrainedModel",
        ]
    )
    _import_structure["models.megatron_bert"].extend(
        [
            "MegatronBertForCausalLM",
            "MegatronBertForMaskedLM",
            "MegatronBertForMultipleChoice",
            "MegatronBertForNextSentencePrediction",
            "MegatronBertForPreTraining",
            "MegatronBertForQuestionAnswering",
            "MegatronBertForSequenceClassification",
            "MegatronBertForTokenClassification",
            "MegatronBertModel",
            "MegatronBertPreTrainedModel",
        ]
    )
    _import_structure["models.mgp_str"].extend(
        [
            "MgpstrForSceneTextRecognition",
            "MgpstrModel",
            "MgpstrPreTrainedModel",
        ]
    )
    _import_structure["models.mimi"].extend(
        [
            "MimiModel",
            "MimiPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        [
            "MistralForCausalLM",
            "MistralForQuestionAnswering",
            "MistralForSequenceClassification",
            "MistralForTokenClassification",
            "MistralModel",
            "MistralPreTrainedModel",
        ]
    )
    _import_structure["models.mixtral"].extend(
        [
            "MixtralForCausalLM",
            "MixtralForQuestionAnswering",
            "MixtralForSequenceClassification",
            "MixtralForTokenClassification",
            "MixtralModel",
            "MixtralPreTrainedModel",
        ]
    )
    _import_structure["models.mllama"].extend(
        [
            "MllamaForCausalLM",
            "MllamaForConditionalGeneration",
            "MllamaPreTrainedModel",
            "MllamaProcessor",
            "MllamaTextModel",
            "MllamaVisionModel",
        ]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "MobileBertForMaskedLM",
            "MobileBertForMultipleChoice",
            "MobileBertForNextSentencePrediction",
            "MobileBertForPreTraining",
            "MobileBertForQuestionAnswering",
            "MobileBertForSequenceClassification",
            "MobileBertForTokenClassification",
            "MobileBertModel",
            "MobileBertPreTrainedModel",
            "load_tf_weights_in_mobilebert",
        ]
    )
    _import_structure["models.mobilenet_v1"].extend(
        [
            "MobileNetV1ForImageClassification",
            "MobileNetV1Model",
            "MobileNetV1PreTrainedModel",
            "load_tf_weights_in_mobilenet_v1",
        ]
    )
    _import_structure["models.mobilenet_v2"].extend(
        [
            "MobileNetV2ForImageClassification",
            "MobileNetV2ForSemanticSegmentation",
            "MobileNetV2Model",
            "MobileNetV2PreTrainedModel",
            "load_tf_weights_in_mobilenet_v2",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "MobileViTForImageClassification",
            "MobileViTForSemanticSegmentation",
            "MobileViTModel",
            "MobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevitv2"].extend(
        [
            "MobileViTV2ForImageClassification",
            "MobileViTV2ForSemanticSegmentation",
            "MobileViTV2Model",
            "MobileViTV2PreTrainedModel",
        ]
    )
    _import_structure["models.moshi"].extend(
        [
            "MoshiForCausalLM",
            "MoshiForConditionalGeneration",
            "MoshiModel",
            "MoshiPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "MPNetForMaskedLM",
            "MPNetForMultipleChoice",
            "MPNetForQuestionAnswering",
            "MPNetForSequenceClassification",
            "MPNetForTokenClassification",
            "MPNetModel",
            "MPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mpt"].extend(
        [
            "MptForCausalLM",
            "MptForQuestionAnswering",
            "MptForSequenceClassification",
            "MptForTokenClassification",
            "MptModel",
            "MptPreTrainedModel",
        ]
    )
    _import_structure["models.mra"].extend(
        [
            "MraForMaskedLM",
            "MraForMultipleChoice",
            "MraForQuestionAnswering",
            "MraForSequenceClassification",
            "MraForTokenClassification",
            "MraModel",
            "MraPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(
        [
            "MT5EncoderModel",
            "MT5ForConditionalGeneration",
            "MT5ForQuestionAnswering",
            "MT5ForSequenceClassification",
            "MT5ForTokenClassification",
            "MT5Model",
            "MT5PreTrainedModel",
        ]
    )
    _import_structure["models.musicgen"].extend(
        [
            "MusicgenForCausalLM",
            "MusicgenForConditionalGeneration",
            "MusicgenModel",
            "MusicgenPreTrainedModel",
            "MusicgenProcessor",
        ]
    )
    _import_structure["models.musicgen_melody"].extend(
        [
            "MusicgenMelodyForCausalLM",
            "MusicgenMelodyForConditionalGeneration",
            "MusicgenMelodyModel",
            "MusicgenMelodyPreTrainedModel",
        ]
    )
    _import_structure["models.mvp"].extend(
        [
            "MvpForCausalLM",
            "MvpForConditionalGeneration",
            "MvpForQuestionAnswering",
            "MvpForSequenceClassification",
            "MvpModel",
            "MvpPreTrainedModel",
        ]
    )
    _import_structure["models.nemotron"].extend(
        [
            "NemotronForCausalLM",
            "NemotronForQuestionAnswering",
            "NemotronForSequenceClassification",
            "NemotronForTokenClassification",
            "NemotronModel",
            "NemotronPreTrainedModel",
        ]
    )
    _import_structure["models.nllb_moe"].extend(
        [
            "NllbMoeForConditionalGeneration",
            "NllbMoeModel",
            "NllbMoePreTrainedModel",
            "NllbMoeSparseMLP",
            "NllbMoeTop2Router",
        ]
    )
    _import_structure["models.nystromformer"].extend(
        [
            "NystromformerForMaskedLM",
            "NystromformerForMultipleChoice",
            "NystromformerForQuestionAnswering",
            "NystromformerForSequenceClassification",
            "NystromformerForTokenClassification",
            "NystromformerModel",
            "NystromformerPreTrainedModel",
        ]
    )
    _import_structure["models.olmo"].extend(
        [
            "OlmoForCausalLM",
            "OlmoModel",
            "OlmoPreTrainedModel",
        ]
    )
    _import_structure["models.olmoe"].extend(
        [
            "OlmoeForCausalLM",
            "OlmoeModel",
            "OlmoePreTrainedModel",
        ]
    )
    _import_structure["models.omdet_turbo"].extend(
        [
            "OmDetTurboForObjectDetection",
            "OmDetTurboPreTrainedModel",
        ]
    )
    _import_structure["models.oneformer"].extend(
        [
            "OneFormerForUniversalSegmentation",
            "OneFormerModel",
            "OneFormerPreTrainedModel",
        ]
    )
    _import_structure["models.openai"].extend(
        [
            "OpenAIGPTDoubleHeadsModel",
            "OpenAIGPTForSequenceClassification",
            "OpenAIGPTLMHeadModel",
            "OpenAIGPTModel",
            "OpenAIGPTPreTrainedModel",
            "load_tf_weights_in_openai_gpt",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "OPTForCausalLM",
            "OPTForQuestionAnswering",
            "OPTForSequenceClassification",
            "OPTModel",
            "OPTPreTrainedModel",
        ]
    )
    _import_structure["models.owlv2"].extend(
        [
            "Owlv2ForObjectDetection",
            "Owlv2Model",
            "Owlv2PreTrainedModel",
            "Owlv2TextModel",
            "Owlv2VisionModel",
        ]
    )
    _import_structure["models.owlvit"].extend(
        [
            "OwlViTForObjectDetection",
            "OwlViTModel",
            "OwlViTPreTrainedModel",
            "OwlViTTextModel",
            "OwlViTVisionModel",
        ]
    )
    _import_structure["models.paligemma"].extend(
        [
            "PaliGemmaForConditionalGeneration",
            "PaliGemmaPreTrainedModel",
            "PaliGemmaProcessor",
        ]
    )
    _import_structure["models.patchtsmixer"].extend(
        [
            "PatchTSMixerForPrediction",
            "PatchTSMixerForPretraining",
            "PatchTSMixerForRegression",
            "PatchTSMixerForTimeSeriesClassification",
            "PatchTSMixerModel",
            "PatchTSMixerPreTrainedModel",
        ]
    )
    _import_structure["models.patchtst"].extend(
        [
            "PatchTSTForClassification",
            "PatchTSTForPrediction",
            "PatchTSTForPretraining",
            "PatchTSTForRegression",
            "PatchTSTModel",
            "PatchTSTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "PegasusForCausalLM",
            "PegasusForConditionalGeneration",
            "PegasusModel",
            "PegasusPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus_x"].extend(
        [
            "PegasusXForConditionalGeneration",
            "PegasusXModel",
            "PegasusXPreTrainedModel",
        ]
    )
    _import_structure["models.perceiver"].extend(
        [
            "PerceiverForImageClassificationConvProcessing",
            "PerceiverForImageClassificationFourier",
            "PerceiverForImageClassificationLearned",
            "PerceiverForMaskedLM",
            "PerceiverForMultimodalAutoencoding",
            "PerceiverForOpticalFlow",
            "PerceiverForSequenceClassification",
            "PerceiverModel",
            "PerceiverPreTrainedModel",
        ]
    )
    _import_structure["models.persimmon"].extend(
        [
            "PersimmonForCausalLM",
            "PersimmonForSequenceClassification",
            "PersimmonForTokenClassification",
            "PersimmonModel",
            "PersimmonPreTrainedModel",
        ]
    )
    _import_structure["models.phi"].extend(
        [
            "PhiForCausalLM",
            "PhiForSequenceClassification",
            "PhiForTokenClassification",
            "PhiModel",
            "PhiPreTrainedModel",
        ]
    )
    _import_structure["models.phi3"].extend(
        [
            "Phi3ForCausalLM",
            "Phi3ForSequenceClassification",
            "Phi3ForTokenClassification",
            "Phi3Model",
            "Phi3PreTrainedModel",
        ]
    )
    _import_structure["models.phimoe"].extend(
        [
            "PhimoeForCausalLM",
            "PhimoeForSequenceClassification",
            "PhimoeModel",
            "PhimoePreTrainedModel",
        ]
    )
    _import_structure["models.pix2struct"].extend(
        [
            "Pix2StructForConditionalGeneration",
            "Pix2StructPreTrainedModel",
            "Pix2StructTextModel",
            "Pix2StructVisionModel",
        ]
    )
    _import_structure["models.pixtral"].extend(["PixtralPreTrainedModel", "PixtralVisionModel"])
    _import_structure["models.plbart"].extend(
        [
            "PLBartForCausalLM",
            "PLBartForConditionalGeneration",
            "PLBartForSequenceClassification",
            "PLBartModel",
            "PLBartPreTrainedModel",
        ]
    )
    _import_structure["models.poolformer"].extend(
        [
            "PoolFormerForImageClassification",
            "PoolFormerModel",
            "PoolFormerPreTrainedModel",
        ]
    )
    _import_structure["models.pop2piano"].extend(
        [
            "Pop2PianoForConditionalGeneration",
            "Pop2PianoPreTrainedModel",
        ]
    )
    _import_structure["models.prophetnet"].extend(
        [
            "ProphetNetDecoder",
            "ProphetNetEncoder",
            "ProphetNetForCausalLM",
            "ProphetNetForConditionalGeneration",
            "ProphetNetModel",
            "ProphetNetPreTrainedModel",
        ]
    )
    _import_structure["models.pvt"].extend(
        [
            "PvtForImageClassification",
            "PvtModel",
            "PvtPreTrainedModel",
        ]
    )
    _import_structure["models.pvt_v2"].extend(
        [
            "PvtV2Backbone",
            "PvtV2ForImageClassification",
            "PvtV2Model",
            "PvtV2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2"].extend(
        [
            "Qwen2ForCausalLM",
            "Qwen2ForQuestionAnswering",
            "Qwen2ForSequenceClassification",
            "Qwen2ForTokenClassification",
            "Qwen2Model",
            "Qwen2PreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_audio"].extend(
        [
            "Qwen2AudioEncoder",
            "Qwen2AudioForConditionalGeneration",
            "Qwen2AudioPreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_moe"].extend(
        [
            "Qwen2MoeForCausalLM",
            "Qwen2MoeForQuestionAnswering",
            "Qwen2MoeForSequenceClassification",
            "Qwen2MoeForTokenClassification",
            "Qwen2MoeModel",
            "Qwen2MoePreTrainedModel",
        ]
    )
    _import_structure["models.qwen2_vl"].extend(
        [
            "Qwen2VLForConditionalGeneration",
            "Qwen2VLModel",
            "Qwen2VLPreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(
        [
            "RagModel",
            "RagPreTrainedModel",
            "RagSequenceForGeneration",
            "RagTokenForGeneration",
        ]
    )
    _import_structure["models.recurrent_gemma"].extend(
        [
            "RecurrentGemmaForCausalLM",
            "RecurrentGemmaModel",
            "RecurrentGemmaPreTrainedModel",
        ]
    )
    _import_structure["models.reformer"].extend(
        [
            "ReformerForMaskedLM",
            "ReformerForQuestionAnswering",
            "ReformerForSequenceClassification",
            "ReformerModel",
            "ReformerModelWithLMHead",
            "ReformerPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "RegNetForImageClassification",
            "RegNetModel",
            "RegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "RemBertForCausalLM",
            "RemBertForMaskedLM",
            "RemBertForMultipleChoice",
            "RemBertForQuestionAnswering",
            "RemBertForSequenceClassification",
            "RemBertForTokenClassification",
            "RemBertModel",
            "RemBertPreTrainedModel",
            "load_tf_weights_in_rembert",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "ResNetBackbone",
            "ResNetForImageClassification",
            "ResNetModel",
            "ResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "RobertaForCausalLM",
            "RobertaForMaskedLM",
            "RobertaForMultipleChoice",
            "RobertaForQuestionAnswering",
            "RobertaForSequenceClassification",
            "RobertaForTokenClassification",
            "RobertaModel",
            "RobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "RobertaPreLayerNormForCausalLM",
            "RobertaPreLayerNormForMaskedLM",
            "RobertaPreLayerNormForMultipleChoice",
            "RobertaPreLayerNormForQuestionAnswering",
            "RobertaPreLayerNormForSequenceClassification",
            "RobertaPreLayerNormForTokenClassification",
            "RobertaPreLayerNormModel",
            "RobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roc_bert"].extend(
        [
            "RoCBertForCausalLM",
            "RoCBertForMaskedLM",
            "RoCBertForMultipleChoice",
            "RoCBertForPreTraining",
            "RoCBertForQuestionAnswering",
            "RoCBertForSequenceClassification",
            "RoCBertForTokenClassification",
            "RoCBertModel",
            "RoCBertPreTrainedModel",
            "load_tf_weights_in_roc_bert",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "RoFormerForCausalLM",
            "RoFormerForMaskedLM",
            "RoFormerForMultipleChoice",
            "RoFormerForQuestionAnswering",
            "RoFormerForSequenceClassification",
            "RoFormerForTokenClassification",
            "RoFormerModel",
            "RoFormerPreTrainedModel",
            "load_tf_weights_in_roformer",
        ]
    )
    _import_structure["models.rt_detr"].extend(
        [
            "RTDetrForObjectDetection",
            "RTDetrModel",
            "RTDetrPreTrainedModel",
            "RTDetrResNetBackbone",
            "RTDetrResNetPreTrainedModel",
        ]
    )
    _import_structure["models.rwkv"].extend(
        [
            "RwkvForCausalLM",
            "RwkvModel",
            "RwkvPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "SamModel",
            "SamPreTrainedModel",
        ]
    )
    _import_structure["models.seamless_m4t"].extend(
        [
            "SeamlessM4TCodeHifiGan",
            "SeamlessM4TForSpeechToSpeech",
            "SeamlessM4TForSpeechToText",
            "SeamlessM4TForTextToSpeech",
            "SeamlessM4TForTextToText",
            "SeamlessM4THifiGan",
            "SeamlessM4TModel",
            "SeamlessM4TPreTrainedModel",
            "SeamlessM4TTextToUnitForConditionalGeneration",
            "SeamlessM4TTextToUnitModel",
        ]
    )
    _import_structure["models.seamless_m4t_v2"].extend(
        [
            "SeamlessM4Tv2ForSpeechToSpeech",
            "SeamlessM4Tv2ForSpeechToText",
            "SeamlessM4Tv2ForTextToSpeech",
            "SeamlessM4Tv2ForTextToText",
            "SeamlessM4Tv2Model",
            "SeamlessM4Tv2PreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "SegformerDecodeHead",
            "SegformerForImageClassification",
            "SegformerForSemanticSegmentation",
            "SegformerModel",
            "SegformerPreTrainedModel",
        ]
    )
    _import_structure["models.seggpt"].extend(
        [
            "SegGptForImageSegmentation",
            "SegGptModel",
            "SegGptPreTrainedModel",
        ]
    )
    _import_structure["models.sew"].extend(
        [
            "SEWForCTC",
            "SEWForSequenceClassification",
            "SEWModel",
            "SEWPreTrainedModel",
        ]
    )
    _import_structure["models.sew_d"].extend(
        [
            "SEWDForCTC",
            "SEWDForSequenceClassification",
            "SEWDModel",
            "SEWDPreTrainedModel",
        ]
    )
    _import_structure["models.siglip"].extend(
        [
            "SiglipForImageClassification",
            "SiglipModel",
            "SiglipPreTrainedModel",
            "SiglipTextModel",
            "SiglipVisionModel",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].extend(["SpeechEncoderDecoderModel"])
    _import_structure["models.speech_to_text"].extend(
        [
            "Speech2TextForConditionalGeneration",
            "Speech2TextModel",
            "Speech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.speecht5"].extend(
        [
            "SpeechT5ForSpeechToSpeech",
            "SpeechT5ForSpeechToText",
            "SpeechT5ForTextToSpeech",
            "SpeechT5HifiGan",
            "SpeechT5Model",
            "SpeechT5PreTrainedModel",
        ]
    )
    _import_structure["models.splinter"].extend(
        [
            "SplinterForPreTraining",
            "SplinterForQuestionAnswering",
            "SplinterModel",
            "SplinterPreTrainedModel",
        ]
    )
    _import_structure["models.squeezebert"].extend(
        [
            "SqueezeBertForMaskedLM",
            "SqueezeBertForMultipleChoice",
            "SqueezeBertForQuestionAnswering",
            "SqueezeBertForSequenceClassification",
            "SqueezeBertForTokenClassification",
            "SqueezeBertModel",
            "SqueezeBertPreTrainedModel",
        ]
    )
    _import_structure["models.stablelm"].extend(
        [
            "StableLmForCausalLM",
            "StableLmForSequenceClassification",
            "StableLmForTokenClassification",
            "StableLmModel",
            "StableLmPreTrainedModel",
        ]
    )
    _import_structure["models.starcoder2"].extend(
        [
            "Starcoder2ForCausalLM",
            "Starcoder2ForSequenceClassification",
            "Starcoder2ForTokenClassification",
            "Starcoder2Model",
            "Starcoder2PreTrainedModel",
        ]
    )
    _import_structure["models.superpoint"].extend(
        [
            "SuperPointForKeypointDetection",
            "SuperPointPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "SwiftFormerForImageClassification",
            "SwiftFormerModel",
            "SwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "SwinBackbone",
            "SwinForImageClassification",
            "SwinForMaskedImageModeling",
            "SwinModel",
            "SwinPreTrainedModel",
        ]
    )
    _import_structure["models.swin2sr"].extend(
        [
            "Swin2SRForImageSuperResolution",
            "Swin2SRModel",
            "Swin2SRPreTrainedModel",
        ]
    )
    _import_structure["models.swinv2"].extend(
        [
            "Swinv2Backbone",
            "Swinv2ForImageClassification",
            "Swinv2ForMaskedImageModeling",
            "Swinv2Model",
            "Swinv2PreTrainedModel",
        ]
    )
    _import_structure["models.switch_transformers"].extend(
        [
            "SwitchTransformersEncoderModel",
            "SwitchTransformersForConditionalGeneration",
            "SwitchTransformersModel",
            "SwitchTransformersPreTrainedModel",
            "SwitchTransformersSparseMLP",
            "SwitchTransformersTop1Router",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "T5EncoderModel",
            "T5ForConditionalGeneration",
            "T5ForQuestionAnswering",
            "T5ForSequenceClassification",
            "T5ForTokenClassification",
            "T5Model",
            "T5PreTrainedModel",
            "load_tf_weights_in_t5",
        ]
    )
    _import_structure["models.table_transformer"].extend(
        [
            "TableTransformerForObjectDetection",
            "TableTransformerModel",
            "TableTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TapasForMaskedLM",
            "TapasForQuestionAnswering",
            "TapasForSequenceClassification",
            "TapasModel",
            "TapasPreTrainedModel",
            "load_tf_weights_in_tapas",
        ]
    )
    _import_structure["models.time_series_transformer"].extend(
        [
            "TimeSeriesTransformerForPrediction",
            "TimeSeriesTransformerModel",
            "TimeSeriesTransformerPreTrainedModel",
        ]
    )
    _import_structure["models.timesformer"].extend(
        [
            "TimesformerForVideoClassification",
            "TimesformerModel",
            "TimesformerPreTrainedModel",
        ]
    )
    _import_structure["models.timm_backbone"].extend(["TimmBackbone"])
    _import_structure["models.trocr"].extend(
        [
            "TrOCRForCausalLM",
            "TrOCRPreTrainedModel",
        ]
    )
    _import_structure["models.tvp"].extend(
        [
            "TvpForVideoGrounding",
            "TvpModel",
            "TvpPreTrainedModel",
        ]
    )
    _import_structure["models.udop"].extend(
        [
            "UdopEncoderModel",
            "UdopForConditionalGeneration",
            "UdopModel",
            "UdopPreTrainedModel",
        ],
    )
    _import_structure["models.umt5"].extend(
        [
            "UMT5EncoderModel",
            "UMT5ForConditionalGeneration",
            "UMT5ForQuestionAnswering",
            "UMT5ForSequenceClassification",
            "UMT5ForTokenClassification",
            "UMT5Model",
            "UMT5PreTrainedModel",
        ]
    )
    _import_structure["models.unispeech"].extend(
        [
            "UniSpeechForCTC",
            "UniSpeechForPreTraining",
            "UniSpeechForSequenceClassification",
            "UniSpeechModel",
            "UniSpeechPreTrainedModel",
        ]
    )
    _import_structure["models.unispeech_sat"].extend(
        [
            "UniSpeechSatForAudioFrameClassification",
            "UniSpeechSatForCTC",
            "UniSpeechSatForPreTraining",
            "UniSpeechSatForSequenceClassification",
            "UniSpeechSatForXVector",
            "UniSpeechSatModel",
            "UniSpeechSatPreTrainedModel",
        ]
    )
    _import_structure["models.univnet"].extend(
        [
            "UnivNetModel",
        ]
    )
    _import_structure["models.upernet"].extend(
        [
            "UperNetForSemanticSegmentation",
            "UperNetPreTrainedModel",
        ]
    )
    _import_structure["models.video_llava"].extend(
        [
            "VideoLlavaForConditionalGeneration",
            "VideoLlavaPreTrainedModel",
            "VideoLlavaProcessor",
        ]
    )
    _import_structure["models.videomae"].extend(
        [
            "VideoMAEForPreTraining",
            "VideoMAEForVideoClassification",
            "VideoMAEModel",
            "VideoMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vilt"].extend(
        [
            "ViltForImageAndTextRetrieval",
            "ViltForImagesAndTextClassification",
            "ViltForMaskedLM",
            "ViltForQuestionAnswering",
            "ViltForTokenClassification",
            "ViltModel",
            "ViltPreTrainedModel",
        ]
    )
    _import_structure["models.vipllava"].extend(
        [
            "VipLlavaForConditionalGeneration",
            "VipLlavaPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["VisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["VisionTextDualEncoderModel"])
    _import_structure["models.visual_bert"].extend(
        [
            "VisualBertForMultipleChoice",
            "VisualBertForPreTraining",
            "VisualBertForQuestionAnswering",
            "VisualBertForRegionToPhraseAlignment",
            "VisualBertForVisualReasoning",
            "VisualBertModel",
            "VisualBertPreTrainedModel",
        ]
    )
    _import_structure["models.vit"].extend(
        [
            "ViTForImageClassification",
            "ViTForMaskedImageModeling",
            "ViTModel",
            "ViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "ViTMAEForPreTraining",
            "ViTMAEModel",
            "ViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.vit_msn"].extend(
        [
            "ViTMSNForImageClassification",
            "ViTMSNModel",
            "ViTMSNPreTrainedModel",
        ]
    )
    _import_structure["models.vitdet"].extend(
        [
            "VitDetBackbone",
            "VitDetModel",
            "VitDetPreTrainedModel",
        ]
    )
    _import_structure["models.vitmatte"].extend(
        [
            "VitMatteForImageMatting",
            "VitMattePreTrainedModel",
        ]
    )
    _import_structure["models.vits"].extend(
        [
            "VitsModel",
            "VitsPreTrainedModel",
        ]
    )
    _import_structure["models.vivit"].extend(
        [
            "VivitForVideoClassification",
            "VivitModel",
            "VivitPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "Wav2Vec2ForAudioFrameClassification",
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForMaskedLM",
            "Wav2Vec2ForPreTraining",
            "Wav2Vec2ForSequenceClassification",
            "Wav2Vec2ForXVector",
            "Wav2Vec2Model",
            "Wav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_bert"].extend(
        [
            "Wav2Vec2BertForAudioFrameClassification",
            "Wav2Vec2BertForCTC",
            "Wav2Vec2BertForSequenceClassification",
            "Wav2Vec2BertForXVector",
            "Wav2Vec2BertModel",
            "Wav2Vec2BertPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2_conformer"].extend(
        [
            "Wav2Vec2ConformerForAudioFrameClassification",
            "Wav2Vec2ConformerForCTC",
            "Wav2Vec2ConformerForPreTraining",
            "Wav2Vec2ConformerForSequenceClassification",
            "Wav2Vec2ConformerForXVector",
            "Wav2Vec2ConformerModel",
            "Wav2Vec2ConformerPreTrainedModel",
        ]
    )
    _import_structure["models.wavlm"].extend(
        [
            "WavLMForAudioFrameClassification",
            "WavLMForCTC",
            "WavLMForSequenceClassification",
            "WavLMForXVector",
            "WavLMModel",
            "WavLMPreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "WhisperForAudioClassification",
            "WhisperForCausalLM",
            "WhisperForConditionalGeneration",
            "WhisperModel",
            "WhisperPreTrainedModel",
        ]
    )
    _import_structure["models.x_clip"].extend(
        [
            "XCLIPModel",
            "XCLIPPreTrainedModel",
            "XCLIPTextModel",
            "XCLIPVisionModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "XGLMForCausalLM",
            "XGLMModel",
            "XGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "XLMForMultipleChoice",
            "XLMForQuestionAnswering",
            "XLMForQuestionAnsweringSimple",
            "XLMForSequenceClassification",
            "XLMForTokenClassification",
            "XLMModel",
            "XLMPreTrainedModel",
            "XLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "XLMRobertaForCausalLM",
            "XLMRobertaForMaskedLM",
            "XLMRobertaForMultipleChoice",
            "XLMRobertaForQuestionAnswering",
            "XLMRobertaForSequenceClassification",
            "XLMRobertaForTokenClassification",
            "XLMRobertaModel",
            "XLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta_xl"].extend(
        [
            "XLMRobertaXLForCausalLM",
            "XLMRobertaXLForMaskedLM",
            "XLMRobertaXLForMultipleChoice",
            "XLMRobertaXLForQuestionAnswering",
            "XLMRobertaXLForSequenceClassification",
            "XLMRobertaXLForTokenClassification",
            "XLMRobertaXLModel",
            "XLMRobertaXLPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "XLNetForMultipleChoice",
            "XLNetForQuestionAnswering",
            "XLNetForQuestionAnsweringSimple",
            "XLNetForSequenceClassification",
            "XLNetForTokenClassification",
            "XLNetLMHeadModel",
            "XLNetModel",
            "XLNetPreTrainedModel",
            "load_tf_weights_in_xlnet",
        ]
    )
    _import_structure["models.xmod"].extend(
        [
            "XmodForCausalLM",
            "XmodForMaskedLM",
            "XmodForMultipleChoice",
            "XmodForQuestionAnswering",
            "XmodForSequenceClassification",
            "XmodForTokenClassification",
            "XmodModel",
            "XmodPreTrainedModel",
        ]
    )
    _import_structure["models.yolos"].extend(
        [
            "YolosForObjectDetection",
            "YolosModel",
            "YolosPreTrainedModel",
        ]
    )
    _import_structure["models.yoso"].extend(
        [
            "YosoForMaskedLM",
            "YosoForMultipleChoice",
            "YosoForQuestionAnswering",
            "YosoForSequenceClassification",
            "YosoForTokenClassification",
            "YosoModel",
            "YosoPreTrainedModel",
        ]
    )
    _import_structure["models.zamba"].extend(
        [
            "ZambaForCausalLM",
            "ZambaForSequenceClassification",
            "ZambaModel",
            "ZambaPreTrainedModel",
        ]
    )
    _import_structure["models.zoedepth"].extend(
        [
            "ZoeDepthForDepthEstimation",
            "ZoeDepthPreTrainedModel",
        ]
    )
    _import_structure["optimization"] = [
        "Adafactor",
        "AdamW",
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_inverse_sqrt_schedule",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
        "get_wsd_schedule",
    ]
    _import_structure["pytorch_utils"] = [
        "Conv1D",
        "apply_chunking_to_forward",
        "prune_layer",
    ]
    _import_structure["sagemaker"] = []
    _import_structure["time_series_utils"] = []
    _import_structure["trainer"] = ["Trainer"]
    _import_structure["trainer_pt_utils"] = ["torch_distributed_zero_first"]
    _import_structure["trainer_seq2seq"] = ["Seq2SeqTrainer"]

# TensorFlow-backed objects
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_tf_objects

    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    _import_structure["activations_tf"] = []
    _import_structure["benchmark.benchmark_args_tf"] = ["TensorFlowBenchmarkArguments"]
    _import_structure["benchmark.benchmark_tf"] = ["TensorFlowBenchmark"]
    _import_structure["generation"].extend(
        [
            "TFForcedBOSTokenLogitsProcessor",
            "TFForcedEOSTokenLogitsProcessor",
            "TFForceTokensLogitsProcessor",
            "TFGenerationMixin",
            "TFLogitsProcessor",
            "TFLogitsProcessorList",
            "TFLogitsWarper",
            "TFMinLengthLogitsProcessor",
            "TFNoBadWordsLogitsProcessor",
            "TFNoRepeatNGramLogitsProcessor",
            "TFRepetitionPenaltyLogitsProcessor",
            "TFSuppressTokensAtBeginLogitsProcessor",
            "TFSuppressTokensLogitsProcessor",
            "TFTemperatureLogitsWarper",
            "TFTopKLogitsWarper",
            "TFTopPLogitsWarper",
        ]
    )
    _import_structure["keras_callbacks"] = ["KerasMetricCallback", "PushToHubCallback"]
    _import_structure["modeling_tf_outputs"] = []
    _import_structure["modeling_tf_utils"] = [
        "TFPreTrainedModel",
        "TFSequenceSummary",
        "TFSharedEmbeddings",
        "shape_list",
    ]
    # TensorFlow models structure
    _import_structure["models.albert"].extend(
        [
            "TFAlbertForMaskedLM",
            "TFAlbertForMultipleChoice",
            "TFAlbertForPreTraining",
            "TFAlbertForQuestionAnswering",
            "TFAlbertForSequenceClassification",
            "TFAlbertForTokenClassification",
            "TFAlbertMainLayer",
            "TFAlbertModel",
            "TFAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING",
            "TF_MODEL_FOR_MASKED_LM_MAPPING",
            "TF_MODEL_FOR_MASK_GENERATION_MAPPING",
            "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "TF_MODEL_FOR_PRETRAINING_MAPPING",
            "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING",
            "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
            "TF_MODEL_FOR_TEXT_ENCODING_MAPPING",
            "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "TF_MODEL_FOR_VISION_2_SEQ_MAPPING",
            "TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING",
            "TF_MODEL_MAPPING",
            "TF_MODEL_WITH_LM_HEAD_MAPPING",
            "TFAutoModel",
            "TFAutoModelForAudioClassification",
            "TFAutoModelForCausalLM",
            "TFAutoModelForDocumentQuestionAnswering",
            "TFAutoModelForImageClassification",
            "TFAutoModelForMaskedImageModeling",
            "TFAutoModelForMaskedLM",
            "TFAutoModelForMaskGeneration",
            "TFAutoModelForMultipleChoice",
            "TFAutoModelForNextSentencePrediction",
            "TFAutoModelForPreTraining",
            "TFAutoModelForQuestionAnswering",
            "TFAutoModelForSemanticSegmentation",
            "TFAutoModelForSeq2SeqLM",
            "TFAutoModelForSequenceClassification",
            "TFAutoModelForSpeechSeq2Seq",
            "TFAutoModelForTableQuestionAnswering",
            "TFAutoModelForTextEncoding",
            "TFAutoModelForTokenClassification",
            "TFAutoModelForVision2Seq",
            "TFAutoModelForZeroShotImageClassification",
            "TFAutoModelWithLMHead",
        ]
    )
    _import_structure["models.bart"].extend(
        [
            "TFBartForConditionalGeneration",
            "TFBartForSequenceClassification",
            "TFBartModel",
            "TFBartPretrainedModel",
        ]
    )
    _import_structure["models.bert"].extend(
        [
            "TFBertForMaskedLM",
            "TFBertForMultipleChoice",
            "TFBertForNextSentencePrediction",
            "TFBertForPreTraining",
            "TFBertForQuestionAnswering",
            "TFBertForSequenceClassification",
            "TFBertForTokenClassification",
            "TFBertLMHeadModel",
            "TFBertMainLayer",
            "TFBertModel",
            "TFBertPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "TFBlenderbotForConditionalGeneration",
            "TFBlenderbotModel",
            "TFBlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "TFBlenderbotSmallForConditionalGeneration",
            "TFBlenderbotSmallModel",
            "TFBlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.blip"].extend(
        [
            "TFBlipForConditionalGeneration",
            "TFBlipForImageTextRetrieval",
            "TFBlipForQuestionAnswering",
            "TFBlipModel",
            "TFBlipPreTrainedModel",
            "TFBlipTextModel",
            "TFBlipVisionModel",
        ]
    )
    _import_structure["models.camembert"].extend(
        [
            "TFCamembertForCausalLM",
            "TFCamembertForMaskedLM",
            "TFCamembertForMultipleChoice",
            "TFCamembertForQuestionAnswering",
            "TFCamembertForSequenceClassification",
            "TFCamembertForTokenClassification",
            "TFCamembertModel",
            "TFCamembertPreTrainedModel",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "TFCLIPModel",
            "TFCLIPPreTrainedModel",
            "TFCLIPTextModel",
            "TFCLIPVisionModel",
        ]
    )
    _import_structure["models.convbert"].extend(
        [
            "TFConvBertForMaskedLM",
            "TFConvBertForMultipleChoice",
            "TFConvBertForQuestionAnswering",
            "TFConvBertForSequenceClassification",
            "TFConvBertForTokenClassification",
            "TFConvBertModel",
            "TFConvBertPreTrainedModel",
        ]
    )
    _import_structure["models.convnext"].extend(
        [
            "TFConvNextForImageClassification",
            "TFConvNextModel",
            "TFConvNextPreTrainedModel",
        ]
    )
    _import_structure["models.convnextv2"].extend(
        [
            "TFConvNextV2ForImageClassification",
            "TFConvNextV2Model",
            "TFConvNextV2PreTrainedModel",
        ]
    )
    _import_structure["models.ctrl"].extend(
        [
            "TFCTRLForSequenceClassification",
            "TFCTRLLMHeadModel",
            "TFCTRLModel",
            "TFCTRLPreTrainedModel",
        ]
    )
    _import_structure["models.cvt"].extend(
        [
            "TFCvtForImageClassification",
            "TFCvtModel",
            "TFCvtPreTrainedModel",
        ]
    )
    _import_structure["models.data2vec"].extend(
        [
            "TFData2VecVisionForImageClassification",
            "TFData2VecVisionForSemanticSegmentation",
            "TFData2VecVisionModel",
            "TFData2VecVisionPreTrainedModel",
        ]
    )
    _import_structure["models.deberta"].extend(
        [
            "TFDebertaForMaskedLM",
            "TFDebertaForQuestionAnswering",
            "TFDebertaForSequenceClassification",
            "TFDebertaForTokenClassification",
            "TFDebertaModel",
            "TFDebertaPreTrainedModel",
        ]
    )
    _import_structure["models.deberta_v2"].extend(
        [
            "TFDebertaV2ForMaskedLM",
            "TFDebertaV2ForMultipleChoice",
            "TFDebertaV2ForQuestionAnswering",
            "TFDebertaV2ForSequenceClassification",
            "TFDebertaV2ForTokenClassification",
            "TFDebertaV2Model",
            "TFDebertaV2PreTrainedModel",
        ]
    )
    _import_structure["models.deit"].extend(
        [
            "TFDeiTForImageClassification",
            "TFDeiTForImageClassificationWithTeacher",
            "TFDeiTForMaskedImageModeling",
            "TFDeiTModel",
            "TFDeiTPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.efficientformer"].extend(
        [
            "TFEfficientFormerForImageClassification",
            "TFEfficientFormerForImageClassificationWithTeacher",
            "TFEfficientFormerModel",
            "TFEfficientFormerPreTrainedModel",
        ]
    )
    _import_structure["models.deprecated.transfo_xl"].extend(
        [
            "TFAdaptiveEmbedding",
            "TFTransfoXLForSequenceClassification",
            "TFTransfoXLLMHeadModel",
            "TFTransfoXLMainLayer",
            "TFTransfoXLModel",
            "TFTransfoXLPreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "TFDistilBertForMaskedLM",
            "TFDistilBertForMultipleChoice",
            "TFDistilBertForQuestionAnswering",
            "TFDistilBertForSequenceClassification",
            "TFDistilBertForTokenClassification",
            "TFDistilBertMainLayer",
            "TFDistilBertModel",
            "TFDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.dpr"].extend(
        [
            "TFDPRContextEncoder",
            "TFDPRPretrainedContextEncoder",
            "TFDPRPretrainedQuestionEncoder",
            "TFDPRPretrainedReader",
            "TFDPRQuestionEncoder",
            "TFDPRReader",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "TFElectraForMaskedLM",
            "TFElectraForMultipleChoice",
            "TFElectraForPreTraining",
            "TFElectraForQuestionAnswering",
            "TFElectraForSequenceClassification",
            "TFElectraForTokenClassification",
            "TFElectraModel",
            "TFElectraPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("TFEncoderDecoderModel")
    _import_structure["models.esm"].extend(
        [
            "TFEsmForMaskedLM",
            "TFEsmForSequenceClassification",
            "TFEsmForTokenClassification",
            "TFEsmModel",
            "TFEsmPreTrainedModel",
        ]
    )
    _import_structure["models.flaubert"].extend(
        [
            "TFFlaubertForMultipleChoice",
            "TFFlaubertForQuestionAnsweringSimple",
            "TFFlaubertForSequenceClassification",
            "TFFlaubertForTokenClassification",
            "TFFlaubertModel",
            "TFFlaubertPreTrainedModel",
            "TFFlaubertWithLMHeadModel",
        ]
    )
    _import_structure["models.funnel"].extend(
        [
            "TFFunnelBaseModel",
            "TFFunnelForMaskedLM",
            "TFFunnelForMultipleChoice",
            "TFFunnelForPreTraining",
            "TFFunnelForQuestionAnswering",
            "TFFunnelForSequenceClassification",
            "TFFunnelForTokenClassification",
            "TFFunnelModel",
            "TFFunnelPreTrainedModel",
        ]
    )
    _import_structure["models.gpt2"].extend(
        [
            "TFGPT2DoubleHeadsModel",
            "TFGPT2ForSequenceClassification",
            "TFGPT2LMHeadModel",
            "TFGPT2MainLayer",
            "TFGPT2Model",
            "TFGPT2PreTrainedModel",
        ]
    )
    _import_structure["models.gptj"].extend(
        [
            "TFGPTJForCausalLM",
            "TFGPTJForQuestionAnswering",
            "TFGPTJForSequenceClassification",
            "TFGPTJModel",
            "TFGPTJPreTrainedModel",
        ]
    )
    _import_structure["models.groupvit"].extend(
        [
            "TFGroupViTModel",
            "TFGroupViTPreTrainedModel",
            "TFGroupViTTextModel",
            "TFGroupViTVisionModel",
        ]
    )
    _import_structure["models.hubert"].extend(
        [
            "TFHubertForCTC",
            "TFHubertModel",
            "TFHubertPreTrainedModel",
        ]
    )

    _import_structure["models.idefics"].extend(
        [
            "TFIdeficsForVisionText2Text",
            "TFIdeficsModel",
            "TFIdeficsPreTrainedModel",
        ]
    )

    _import_structure["models.layoutlm"].extend(
        [
            "TFLayoutLMForMaskedLM",
            "TFLayoutLMForQuestionAnswering",
            "TFLayoutLMForSequenceClassification",
            "TFLayoutLMForTokenClassification",
            "TFLayoutLMMainLayer",
            "TFLayoutLMModel",
            "TFLayoutLMPreTrainedModel",
        ]
    )
    _import_structure["models.layoutlmv3"].extend(
        [
            "TFLayoutLMv3ForQuestionAnswering",
            "TFLayoutLMv3ForSequenceClassification",
            "TFLayoutLMv3ForTokenClassification",
            "TFLayoutLMv3Model",
            "TFLayoutLMv3PreTrainedModel",
        ]
    )
    _import_structure["models.led"].extend(["TFLEDForConditionalGeneration", "TFLEDModel", "TFLEDPreTrainedModel"])
    _import_structure["models.longformer"].extend(
        [
            "TFLongformerForMaskedLM",
            "TFLongformerForMultipleChoice",
            "TFLongformerForQuestionAnswering",
            "TFLongformerForSequenceClassification",
            "TFLongformerForTokenClassification",
            "TFLongformerModel",
            "TFLongformerPreTrainedModel",
        ]
    )
    _import_structure["models.lxmert"].extend(
        [
            "TFLxmertForPreTraining",
            "TFLxmertMainLayer",
            "TFLxmertModel",
            "TFLxmertPreTrainedModel",
            "TFLxmertVisualFeatureEncoder",
        ]
    )
    _import_structure["models.marian"].extend(["TFMarianModel", "TFMarianMTModel", "TFMarianPreTrainedModel"])
    _import_structure["models.mbart"].extend(
        ["TFMBartForConditionalGeneration", "TFMBartModel", "TFMBartPreTrainedModel"]
    )
    _import_structure["models.mistral"].extend(
        ["TFMistralForCausalLM", "TFMistralForSequenceClassification", "TFMistralModel", "TFMistralPreTrainedModel"]
    )
    _import_structure["models.mobilebert"].extend(
        [
            "TFMobileBertForMaskedLM",
            "TFMobileBertForMultipleChoice",
            "TFMobileBertForNextSentencePrediction",
            "TFMobileBertForPreTraining",
            "TFMobileBertForQuestionAnswering",
            "TFMobileBertForSequenceClassification",
            "TFMobileBertForTokenClassification",
            "TFMobileBertMainLayer",
            "TFMobileBertModel",
            "TFMobileBertPreTrainedModel",
        ]
    )
    _import_structure["models.mobilevit"].extend(
        [
            "TFMobileViTForImageClassification",
            "TFMobileViTForSemanticSegmentation",
            "TFMobileViTModel",
            "TFMobileViTPreTrainedModel",
        ]
    )
    _import_structure["models.mpnet"].extend(
        [
            "TFMPNetForMaskedLM",
            "TFMPNetForMultipleChoice",
            "TFMPNetForQuestionAnswering",
            "TFMPNetForSequenceClassification",
            "TFMPNetForTokenClassification",
            "TFMPNetMainLayer",
            "TFMPNetModel",
            "TFMPNetPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["TFMT5EncoderModel", "TFMT5ForConditionalGeneration", "TFMT5Model"])
    _import_structure["models.openai"].extend(
        [
            "TFOpenAIGPTDoubleHeadsModel",
            "TFOpenAIGPTForSequenceClassification",
            "TFOpenAIGPTLMHeadModel",
            "TFOpenAIGPTMainLayer",
            "TFOpenAIGPTModel",
            "TFOpenAIGPTPreTrainedModel",
        ]
    )
    _import_structure["models.opt"].extend(
        [
            "TFOPTForCausalLM",
            "TFOPTModel",
            "TFOPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "TFPegasusForConditionalGeneration",
            "TFPegasusModel",
            "TFPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.rag"].extend(
        [
            "TFRagModel",
            "TFRagPreTrainedModel",
            "TFRagSequenceForGeneration",
            "TFRagTokenForGeneration",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "TFRegNetForImageClassification",
            "TFRegNetModel",
            "TFRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.rembert"].extend(
        [
            "TFRemBertForCausalLM",
            "TFRemBertForMaskedLM",
            "TFRemBertForMultipleChoice",
            "TFRemBertForQuestionAnswering",
            "TFRemBertForSequenceClassification",
            "TFRemBertForTokenClassification",
            "TFRemBertModel",
            "TFRemBertPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "TFResNetForImageClassification",
            "TFResNetModel",
            "TFResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "TFRobertaForCausalLM",
            "TFRobertaForMaskedLM",
            "TFRobertaForMultipleChoice",
            "TFRobertaForQuestionAnswering",
            "TFRobertaForSequenceClassification",
            "TFRobertaForTokenClassification",
            "TFRobertaMainLayer",
            "TFRobertaModel",
            "TFRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "TFRobertaPreLayerNormForCausalLM",
            "TFRobertaPreLayerNormForMaskedLM",
            "TFRobertaPreLayerNormForMultipleChoice",
            "TFRobertaPreLayerNormForQuestionAnswering",
            "TFRobertaPreLayerNormForSequenceClassification",
            "TFRobertaPreLayerNormForTokenClassification",
            "TFRobertaPreLayerNormMainLayer",
            "TFRobertaPreLayerNormModel",
            "TFRobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "TFRoFormerForCausalLM",
            "TFRoFormerForMaskedLM",
            "TFRoFormerForMultipleChoice",
            "TFRoFormerForQuestionAnswering",
            "TFRoFormerForSequenceClassification",
            "TFRoFormerForTokenClassification",
            "TFRoFormerModel",
            "TFRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.sam"].extend(
        [
            "TFSamModel",
            "TFSamPreTrainedModel",
        ]
    )
    _import_structure["models.segformer"].extend(
        [
            "TFSegformerDecodeHead",
            "TFSegformerForImageClassification",
            "TFSegformerForSemanticSegmentation",
            "TFSegformerModel",
            "TFSegformerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_to_text"].extend(
        [
            "TFSpeech2TextForConditionalGeneration",
            "TFSpeech2TextModel",
            "TFSpeech2TextPreTrainedModel",
        ]
    )
    _import_structure["models.swiftformer"].extend(
        [
            "TFSwiftFormerForImageClassification",
            "TFSwiftFormerModel",
            "TFSwiftFormerPreTrainedModel",
        ]
    )
    _import_structure["models.swin"].extend(
        [
            "TFSwinForImageClassification",
            "TFSwinForMaskedImageModeling",
            "TFSwinModel",
            "TFSwinPreTrainedModel",
        ]
    )
    _import_structure["models.t5"].extend(
        [
            "TFT5EncoderModel",
            "TFT5ForConditionalGeneration",
            "TFT5Model",
            "TFT5PreTrainedModel",
        ]
    )
    _import_structure["models.tapas"].extend(
        [
            "TFTapasForMaskedLM",
            "TFTapasForQuestionAnswering",
            "TFTapasForSequenceClassification",
            "TFTapasModel",
            "TFTapasPreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].extend(["TFVisionEncoderDecoderModel"])
    _import_structure["models.vision_text_dual_encoder"].extend(["TFVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(
        [
            "TFViTForImageClassification",
            "TFViTModel",
            "TFViTPreTrainedModel",
        ]
    )
    _import_structure["models.vit_mae"].extend(
        [
            "TFViTMAEForPreTraining",
            "TFViTMAEModel",
            "TFViTMAEPreTrainedModel",
        ]
    )
    _import_structure["models.wav2vec2"].extend(
        [
            "TFWav2Vec2ForCTC",
            "TFWav2Vec2ForSequenceClassification",
            "TFWav2Vec2Model",
            "TFWav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "TFWhisperForConditionalGeneration",
            "TFWhisperModel",
            "TFWhisperPreTrainedModel",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "TFXGLMForCausalLM",
            "TFXGLMModel",
            "TFXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm"].extend(
        [
            "TFXLMForMultipleChoice",
            "TFXLMForQuestionAnsweringSimple",
            "TFXLMForSequenceClassification",
            "TFXLMForTokenClassification",
            "TFXLMMainLayer",
            "TFXLMModel",
            "TFXLMPreTrainedModel",
            "TFXLMWithLMHeadModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "TFXLMRobertaForCausalLM",
            "TFXLMRobertaForMaskedLM",
            "TFXLMRobertaForMultipleChoice",
            "TFXLMRobertaForQuestionAnswering",
            "TFXLMRobertaForSequenceClassification",
            "TFXLMRobertaForTokenClassification",
            "TFXLMRobertaModel",
            "TFXLMRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.xlnet"].extend(
        [
            "TFXLNetForMultipleChoice",
            "TFXLNetForQuestionAnsweringSimple",
            "TFXLNetForSequenceClassification",
            "TFXLNetForTokenClassification",
            "TFXLNetLMHeadModel",
            "TFXLNetMainLayer",
            "TFXLNetModel",
            "TFXLNetPreTrainedModel",
        ]
    )
    _import_structure["optimization_tf"] = [
        "AdamWeightDecay",
        "GradientAccumulator",
        "WarmUp",
        "create_optimizer",
    ]
    _import_structure["tf_utils"] = []


try:
    if not (
        is_librosa_available()
        and is_essentia_available()
        and is_scipy_available()
        and is_torch_available()
        and is_pretty_midi_available()
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import (
        dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects,
    )

    _import_structure["utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects"] = [
        name
        for name in dir(dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects)
        if not name.startswith("_")
    ]
else:
    _import_structure["models.pop2piano"].append("Pop2PianoFeatureExtractor")
    _import_structure["models.pop2piano"].append("Pop2PianoTokenizer")
    _import_structure["models.pop2piano"].append("Pop2PianoProcessor")

try:
    if not is_torchaudio_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import (
        dummy_torchaudio_objects,
    )

    _import_structure["utils.dummy_torchaudio_objects"] = [
        name for name in dir(dummy_torchaudio_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyFeatureExtractor")
    _import_structure["models.musicgen_melody"].append("MusicgenMelodyProcessor")


# FLAX-backed objects
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from transformers.src.transformers.utils import dummy_flax_objects

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    _import_structure["generation"].extend(
        [
            "FlaxForcedBOSTokenLogitsProcessor",
            "FlaxForcedEOSTokenLogitsProcessor",
            "FlaxForceTokensLogitsProcessor",
            "FlaxGenerationMixin",
            "FlaxLogitsProcessor",
            "FlaxLogitsProcessorList",
            "FlaxLogitsWarper",
            "FlaxMinLengthLogitsProcessor",
            "FlaxTemperatureLogitsWarper",
            "FlaxSuppressTokensAtBeginLogitsProcessor",
            "FlaxSuppressTokensLogitsProcessor",
            "FlaxTopKLogitsWarper",
            "FlaxTopPLogitsWarper",
            "FlaxWhisperTimeStampLogitsProcessor",
        ]
    )
    _import_structure["modeling_flax_outputs"] = []
    _import_structure["modeling_flax_utils"] = ["FlaxPreTrainedModel"]
    _import_structure["models.albert"].extend(
        [
            "FlaxAlbertForMaskedLM",
            "FlaxAlbertForMultipleChoice",
            "FlaxAlbertForPreTraining",
            "FlaxAlbertForQuestionAnswering",
            "FlaxAlbertForSequenceClassification",
            "FlaxAlbertForTokenClassification",
            "FlaxAlbertModel",
            "FlaxAlbertPreTrainedModel",
        ]
    )
    _import_structure["models.auto"].extend(
        [
            "FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_CAUSAL_LM_MAPPING",
            "FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_MASKED_LM_MAPPING",
            "FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
            "FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
            "FLAX_MODEL_FOR_PRETRAINING_MAPPING",
            "FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
            "FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
            "FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING",
            "FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
            "FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING",
            "FLAX_MODEL_MAPPING",
            "FlaxAutoModel",
            "FlaxAutoModelForCausalLM",
            "FlaxAutoModelForImageClassification",
            "FlaxAutoModelForMaskedLM",
            "FlaxAutoModelForMultipleChoice",
            "FlaxAutoModelForNextSentencePrediction",
            "FlaxAutoModelForPreTraining",
            "FlaxAutoModelForQuestionAnswering",
            "FlaxAutoModelForSeq2SeqLM",
            "FlaxAutoModelForSequenceClassification",
            "FlaxAutoModelForSpeechSeq2Seq",
            "FlaxAutoModelForTokenClassification",
            "FlaxAutoModelForVision2Seq",
        ]
    )

    # Flax models structure

    _import_structure["models.bart"].extend(
        [
            "FlaxBartDecoderPreTrainedModel",
            "FlaxBartForCausalLM",
            "FlaxBartForConditionalGeneration",
            "FlaxBartForQuestionAnswering",
            "FlaxBartForSequenceClassification",
            "FlaxBartModel",
            "FlaxBartPreTrainedModel",
        ]
    )
    _import_structure["models.beit"].extend(
        [
            "FlaxBeitForImageClassification",
            "FlaxBeitForMaskedImageModeling",
            "FlaxBeitModel",
            "FlaxBeitPreTrainedModel",
        ]
    )

    _import_structure["models.bert"].extend(
        [
            "FlaxBertForCausalLM",
            "FlaxBertForMaskedLM",
            "FlaxBertForMultipleChoice",
            "FlaxBertForNextSentencePrediction",
            "FlaxBertForPreTraining",
            "FlaxBertForQuestionAnswering",
            "FlaxBertForSequenceClassification",
            "FlaxBertForTokenClassification",
            "FlaxBertModel",
            "FlaxBertPreTrainedModel",
        ]
    )
    _import_structure["models.big_bird"].extend(
        [
            "FlaxBigBirdForCausalLM",
            "FlaxBigBirdForMaskedLM",
            "FlaxBigBirdForMultipleChoice",
            "FlaxBigBirdForPreTraining",
            "FlaxBigBirdForQuestionAnswering",
            "FlaxBigBirdForSequenceClassification",
            "FlaxBigBirdForTokenClassification",
            "FlaxBigBirdModel",
            "FlaxBigBirdPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot"].extend(
        [
            "FlaxBlenderbotForConditionalGeneration",
            "FlaxBlenderbotModel",
            "FlaxBlenderbotPreTrainedModel",
        ]
    )
    _import_structure["models.blenderbot_small"].extend(
        [
            "FlaxBlenderbotSmallForConditionalGeneration",
            "FlaxBlenderbotSmallModel",
            "FlaxBlenderbotSmallPreTrainedModel",
        ]
    )
    _import_structure["models.bloom"].extend(
        [
            "FlaxBloomForCausalLM",
            "FlaxBloomModel",
            "FlaxBloomPreTrainedModel",
        ]
    )
    _import_structure["models.clip"].extend(
        [
            "FlaxCLIPModel",
            "FlaxCLIPPreTrainedModel",
            "FlaxCLIPTextModel",
            "FlaxCLIPTextPreTrainedModel",
            "FlaxCLIPTextModelWithProjection",
            "FlaxCLIPVisionModel",
            "FlaxCLIPVisionPreTrainedModel",
        ]
    )
    _import_structure["models.dinov2"].extend(
        [
            "FlaxDinov2Model",
            "FlaxDinov2ForImageClassification",
            "FlaxDinov2PreTrainedModel",
        ]
    )
    _import_structure["models.distilbert"].extend(
        [
            "FlaxDistilBertForMaskedLM",
            "FlaxDistilBertForMultipleChoice",
            "FlaxDistilBertForQuestionAnswering",
            "FlaxDistilBertForSequenceClassification",
            "FlaxDistilBertForTokenClassification",
            "FlaxDistilBertModel",
            "FlaxDistilBertPreTrainedModel",
        ]
    )
    _import_structure["models.electra"].extend(
        [
            "FlaxElectraForCausalLM",
            "FlaxElectraForMaskedLM",
            "FlaxElectraForMultipleChoice",
            "FlaxElectraForPreTraining",
            "FlaxElectraForQuestionAnswering",
            "FlaxElectraForSequenceClassification",
            "FlaxElectraForTokenClassification",
            "FlaxElectraModel",
            "FlaxElectraPreTrainedModel",
        ]
    )
    _import_structure["models.encoder_decoder"].append("FlaxEncoderDecoderModel")
    _import_structure["models.gpt2"].extend(["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"])
    _import_structure["models.gpt_neo"].extend(
        ["FlaxGPTNeoForCausalLM", "FlaxGPTNeoModel", "FlaxGPTNeoPreTrainedModel"]
    )
    _import_structure["models.gptj"].extend(["FlaxGPTJForCausalLM", "FlaxGPTJModel", "FlaxGPTJPreTrainedModel"])
    _import_structure["models.llama"].extend(["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"])
    _import_structure["models.gemma"].extend(["FlaxGemmaForCausalLM", "FlaxGemmaModel", "FlaxGemmaPreTrainedModel"])
    _import_structure["models.longt5"].extend(
        [
            "FlaxLongT5ForConditionalGeneration",
            "FlaxLongT5Model",
            "FlaxLongT5PreTrainedModel",
        ]
    )
    _import_structure["models.marian"].extend(
        [
            "FlaxMarianModel",
            "FlaxMarianMTModel",
            "FlaxMarianPreTrainedModel",
        ]
    )
    _import_structure["models.mbart"].extend(
        [
            "FlaxMBartForConditionalGeneration",
            "FlaxMBartForQuestionAnswering",
            "FlaxMBartForSequenceClassification",
            "FlaxMBartModel",
            "FlaxMBartPreTrainedModel",
        ]
    )
    _import_structure["models.mistral"].extend(
        [
            "FlaxMistralForCausalLM",
            "FlaxMistralModel",
            "FlaxMistralPreTrainedModel",
        ]
    )
    _import_structure["models.mt5"].extend(["FlaxMT5EncoderModel", "FlaxMT5ForConditionalGeneration", "FlaxMT5Model"])
    _import_structure["models.opt"].extend(
        [
            "FlaxOPTForCausalLM",
            "FlaxOPTModel",
            "FlaxOPTPreTrainedModel",
        ]
    )
    _import_structure["models.pegasus"].extend(
        [
            "FlaxPegasusForConditionalGeneration",
            "FlaxPegasusModel",
            "FlaxPegasusPreTrainedModel",
        ]
    )
    _import_structure["models.regnet"].extend(
        [
            "FlaxRegNetForImageClassification",
            "FlaxRegNetModel",
            "FlaxRegNetPreTrainedModel",
        ]
    )
    _import_structure["models.resnet"].extend(
        [
            "FlaxResNetForImageClassification",
            "FlaxResNetModel",
            "FlaxResNetPreTrainedModel",
        ]
    )
    _import_structure["models.roberta"].extend(
        [
            "FlaxRobertaForCausalLM",
            "FlaxRobertaForMaskedLM",
            "FlaxRobertaForMultipleChoice",
            "FlaxRobertaForQuestionAnswering",
            "FlaxRobertaForSequenceClassification",
            "FlaxRobertaForTokenClassification",
            "FlaxRobertaModel",
            "FlaxRobertaPreTrainedModel",
        ]
    )
    _import_structure["models.roberta_prelayernorm"].extend(
        [
            "FlaxRobertaPreLayerNormForCausalLM",
            "FlaxRobertaPreLayerNormForMaskedLM",
            "FlaxRobertaPreLayerNormForMultipleChoice",
            "FlaxRobertaPreLayerNormForQuestionAnswering",
            "FlaxRobertaPreLayerNormForSequenceClassification",
            "FlaxRobertaPreLayerNormForTokenClassification",
            "FlaxRobertaPreLayerNormModel",
            "FlaxRobertaPreLayerNormPreTrainedModel",
        ]
    )
    _import_structure["models.roformer"].extend(
        [
            "FlaxRoFormerForMaskedLM",
            "FlaxRoFormerForMultipleChoice",
            "FlaxRoFormerForQuestionAnswering",
            "FlaxRoFormerForSequenceClassification",
            "FlaxRoFormerForTokenClassification",
            "FlaxRoFormerModel",
            "FlaxRoFormerPreTrainedModel",
        ]
    )
    _import_structure["models.speech_encoder_decoder"].append("FlaxSpeechEncoderDecoderModel")
    _import_structure["models.t5"].extend(
        [
            "FlaxT5EncoderModel",
            "FlaxT5ForConditionalGeneration",
            "FlaxT5Model",
            "FlaxT5PreTrainedModel",
        ]
    )
    _import_structure["models.vision_encoder_decoder"].append("FlaxVisionEncoderDecoderModel")
    _import_structure["models.vision_text_dual_encoder"].extend(["FlaxVisionTextDualEncoderModel"])
    _import_structure["models.vit"].extend(["FlaxViTForImageClassification", "FlaxViTModel", "FlaxViTPreTrainedModel"])
    _import_structure["models.wav2vec2"].extend(
        [
            "FlaxWav2Vec2ForCTC",
            "FlaxWav2Vec2ForPreTraining",
            "FlaxWav2Vec2Model",
            "FlaxWav2Vec2PreTrainedModel",
        ]
    )
    _import_structure["models.whisper"].extend(
        [
            "FlaxWhisperForConditionalGeneration",
            "FlaxWhisperModel",
            "FlaxWhisperPreTrainedModel",
            "FlaxWhisperForAudioClassification",
        ]
    )
    _import_structure["models.xglm"].extend(
        [
            "FlaxXGLMForCausalLM",
            "FlaxXGLMModel",
            "FlaxXGLMPreTrainedModel",
        ]
    )
    _import_structure["models.xlm_roberta"].extend(
        [
            "FlaxXLMRobertaForMaskedLM",
            "FlaxXLMRobertaForMultipleChoice",
            "FlaxXLMRobertaForQuestionAnswering",
            "FlaxXLMRobertaForSequenceClassification",
            "FlaxXLMRobertaForTokenClassification",
            "FlaxXLMRobertaModel",
            "FlaxXLMRobertaForCausalLM",
            "FlaxXLMRobertaPreTrainedModel",
        ]
    )


# Direct imports for type-checking
if TYPE_CHECKING:
    # Configuration
    # Agents
    from transformers.src.transformers.agents import (
        Agent,
        CodeAgent,
        HfApiEngine,
        ManagedAgent,
        PipelineTool,
        ReactAgent,
        ReactCodeAgent,
        ReactJsonAgent,
        Tool,
        Toolbox,
        ToolCollection,
        TransformersEngine,
        launch_gradio_demo,
        load_tool,
        stream_to_gradio,
        tool,
    )
    from transformers.src.transformers.configuration_utils import PretrainedConfig

    # Data
    from transformers.src.transformers.data import (
        DataProcessor,
        InputExample,
        InputFeatures,
        SingleSentenceClassificationProcessor,
        SquadExample,
        SquadFeatures,
        SquadV1Processor,
        SquadV2Processor,
        glue_compute_metrics,
        glue_convert_examples_to_features,
        glue_output_modes,
        glue_processors,
        glue_tasks_num_labels,
        squad_convert_examples_to_features,
        xnli_compute_metrics,
        xnli_output_modes,
        xnli_processors,
        xnli_tasks_num_labels,
    )
    from transformers.src.transformers.data.data_collator import (
        DataCollator,
        DataCollatorForLanguageModeling,
        DataCollatorForPermutationLanguageModeling,
        DataCollatorForSeq2Seq,
        DataCollatorForSOP,
        DataCollatorForTokenClassification,
        DataCollatorForWholeWordMask,
        DataCollatorWithFlattening,
        DataCollatorWithPadding,
        DefaultDataCollator,
        default_data_collator,
    )
    from transformers.src.transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

    # Feature Extractor
    from transformers.src.transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin

    # Generation
    from transformers.src.transformers.generation import GenerationConfig, TextIteratorStreamer, TextStreamer, WatermarkingConfig
    from transformers.src.transformers.hf_argparser import HfArgumentParser

    # Integrations
    from transformers.src.transformers.integrations import (
        is_clearml_available,
        is_comet_available,
        is_dvclive_available,
        is_neptune_available,
        is_optuna_available,
        is_ray_available,
        is_ray_tune_available,
        is_sigopt_available,
        is_tensorboard_available,
        is_wandb_available,
    )

    # Model Cards
    from transformers.src.transformers.modelcard import ModelCard

    # TF 2.0 <=> PyTorch conversion utilities
    from transformers.src.transformers.modeling_tf_pytorch_utils import (
        convert_tf_weight_name_to_pt_weight_name,
        load_pytorch_checkpoint_in_tf2_model,
        load_pytorch_model_in_tf2_model,
        load_pytorch_weights_in_tf2_model,
        load_tf2_checkpoint_in_pytorch_model,
        load_tf2_model_in_pytorch_model,
        load_tf2_weights_in_pytorch_model,
    )
    from transformers.src.transformers.models.albert import AlbertConfig
    from transformers.src.transformers.models.align import (
        AlignConfig,
        AlignProcessor,
        AlignTextConfig,
        AlignVisionConfig,
    )
    from transformers.src.transformers.models.altclip import (
        AltCLIPConfig,
        AltCLIPProcessor,
        AltCLIPTextConfig,
        AltCLIPVisionConfig,
    )
    from transformers.src.transformers.models.audio_spectrogram_transformer import (
        ASTConfig,
        ASTFeatureExtractor,
    )
    from transformers.src.transformers.models.auto import (
        CONFIG_MAPPING,
        FEATURE_EXTRACTOR_MAPPING,
        IMAGE_PROCESSOR_MAPPING,
        MODEL_NAMES_MAPPING,
        PROCESSOR_MAPPING,
        TOKENIZER_MAPPING,
        AutoConfig,
        AutoFeatureExtractor,
        AutoImageProcessor,
        AutoProcessor,
        AutoTokenizer,
    )
    from transformers.src.transformers.models.autoformer import (
        AutoformerConfig,
    )
    from transformers.src.transformers.models.bark import (
        BarkCoarseConfig,
        BarkConfig,
        BarkFineConfig,
        BarkProcessor,
        BarkSemanticConfig,
    )
    from transformers.src.transformers.models.bart import BartConfig, BartTokenizer
    from transformers.src.transformers.models.beit import BeitConfig
    from transformers.src.transformers.models.bert import (
        BasicTokenizer,
        BertConfig,
        BertTokenizer,
        WordpieceTokenizer,
    )
    from transformers.src.transformers.models.bert_generation import BertGenerationConfig
    from transformers.src.transformers.models.bert_japanese import (
        BertJapaneseTokenizer,
        CharacterTokenizer,
        MecabTokenizer,
    )
    from transformers.src.transformers.models.bertweet import BertweetTokenizer
    from transformers.src.transformers.models.big_bird import BigBirdConfig
    from transformers.src.transformers.models.bigbird_pegasus import (
        BigBirdPegasusConfig,
    )
    from transformers.src.transformers.models.biogpt import (
        BioGptConfig,
        BioGptTokenizer,
    )
    from transformers.src.transformers.models.bit import BitConfig
    from transformers.src.transformers.models.blenderbot import (
        BlenderbotConfig,
        BlenderbotTokenizer,
    )
    from transformers.src.transformers.models.blenderbot_small import (
        BlenderbotSmallConfig,
        BlenderbotSmallTokenizer,
    )
    from transformers.src.transformers.models.blip import (
        BlipConfig,
        BlipProcessor,
        BlipTextConfig,
        BlipVisionConfig,
    )
    from transformers.src.transformers.models.blip_2 import (
        Blip2Config,
        Blip2Processor,
        Blip2QFormerConfig,
        Blip2VisionConfig,
    )
    from transformers.src.transformers.models.bloom import BloomConfig
    from transformers.src.transformers.models.bridgetower import (
        BridgeTowerConfig,
        BridgeTowerProcessor,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    from transformers.src.transformers.models.bros import (
        BrosConfig,
        BrosProcessor,
    )
    from transformers.src.transformers.models.byt5 import ByT5Tokenizer
    from transformers.src.transformers.models.camembert import (
        CamembertConfig,
    )
    from transformers.src.transformers.models.canine import (
        CanineConfig,
        CanineTokenizer,
    )
    from transformers.src.transformers.models.chameleon import (
        ChameleonConfig,
        ChameleonProcessor,
        ChameleonVQVAEConfig,
    )
    from transformers.src.transformers.models.chinese_clip import (
        ChineseCLIPConfig,
        ChineseCLIPProcessor,
        ChineseCLIPTextConfig,
        ChineseCLIPVisionConfig,
    )
    from transformers.src.transformers.models.clap import (
        ClapAudioConfig,
        ClapConfig,
        ClapProcessor,
        ClapTextConfig,
    )
    from transformers.src.transformers.models.clip import (
        CLIPConfig,
        CLIPProcessor,
        CLIPTextConfig,
        CLIPTokenizer,
        CLIPVisionConfig,
    )
    from transformers.src.transformers.models.clipseg import (
        CLIPSegConfig,
        CLIPSegProcessor,
        CLIPSegTextConfig,
        CLIPSegVisionConfig,
    )
    from transformers.src.transformers.models.clvp import (
        ClvpConfig,
        ClvpDecoderConfig,
        ClvpEncoderConfig,
        ClvpFeatureExtractor,
        ClvpProcessor,
        ClvpTokenizer,
    )
    from transformers.src.transformers.models.codegen import (
        CodeGenConfig,
        CodeGenTokenizer,
    )
    from transformers.src.transformers.models.cohere import CohereConfig
    from transformers.src.transformers.models.conditional_detr import (
        ConditionalDetrConfig,
    )
    from transformers.src.transformers.models.convbert import (
        ConvBertConfig,
        ConvBertTokenizer,
    )
    from transformers.src.transformers.models.convnext import ConvNextConfig
    from transformers.src.transformers.models.convnextv2 import (
        ConvNextV2Config,
    )
    from transformers.src.transformers.models.cpmant import (
        CpmAntConfig,
        CpmAntTokenizer,
    )
    from transformers.src.transformers.models.ctrl import (
        CTRLConfig,
        CTRLTokenizer,
    )
    from transformers.src.transformers.models.cvt import CvtConfig
    from transformers.src.transformers.models.dac import (
        DacConfig,
        DacFeatureExtractor,
    )
    from transformers.src.transformers.models.data2vec import (
        Data2VecAudioConfig,
        Data2VecTextConfig,
        Data2VecVisionConfig,
    )
    from transformers.src.transformers.models.dbrx import DbrxConfig
    from transformers.src.transformers.models.deberta import (
        DebertaConfig,
        DebertaTokenizer,
    )
    from transformers.src.transformers.models.deberta_v2 import (
        DebertaV2Config,
    )
    from transformers.src.transformers.models.decision_transformer import (
        DecisionTransformerConfig,
    )
    from transformers.src.transformers.models.deformable_detr import (
        DeformableDetrConfig,
    )
    from transformers.src.transformers.models.deit import DeiTConfig
    from transformers.src.transformers.models.deprecated.deta import DetaConfig
    from transformers.src.transformers.models.deprecated.efficientformer import (
        EfficientFormerConfig,
    )
    from transformers.src.transformers.models.deprecated.ernie_m import ErnieMConfig
    from transformers.src.transformers.models.deprecated.gptsan_japanese import (
        GPTSanJapaneseConfig,
        GPTSanJapaneseTokenizer,
    )
    from transformers.src.transformers.models.deprecated.graphormer import GraphormerConfig
    from transformers.src.transformers.models.deprecated.jukebox import (
        JukeboxConfig,
        JukeboxPriorConfig,
        JukeboxTokenizer,
        JukeboxVQVAEConfig,
    )
    from transformers.src.transformers.models.deprecated.mctct import (
        MCTCTConfig,
        MCTCTFeatureExtractor,
        MCTCTProcessor,
    )
    from transformers.src.transformers.models.deprecated.mega import MegaConfig
    from transformers.src.transformers.models.deprecated.mmbt import MMBTConfig
    from transformers.src.transformers.models.deprecated.nat import NatConfig
    from transformers.src.transformers.models.deprecated.nezha import NezhaConfig
    from transformers.src.transformers.models.deprecated.open_llama import (
        OpenLlamaConfig,
    )
    from transformers.src.transformers.models.deprecated.qdqbert import QDQBertConfig
    from transformers.src.transformers.models.deprecated.realm import (
        RealmConfig,
        RealmTokenizer,
    )
    from transformers.src.transformers.models.deprecated.retribert import (
        RetriBertConfig,
        RetriBertTokenizer,
    )
    from transformers.src.transformers.models.deprecated.speech_to_text_2 import (
        Speech2Text2Config,
        Speech2Text2Processor,
        Speech2Text2Tokenizer,
    )
    from transformers.src.transformers.models.deprecated.tapex import TapexTokenizer
    from transformers.src.transformers.models.deprecated.trajectory_transformer import (
        TrajectoryTransformerConfig,
    )
    from transformers.src.transformers.models.deprecated.transfo_xl import (
        TransfoXLConfig,
        TransfoXLCorpus,
        TransfoXLTokenizer,
    )
    from transformers.src.transformers.models.deprecated.tvlt import (
        TvltConfig,
        TvltFeatureExtractor,
        TvltProcessor,
    )
    from transformers.src.transformers.models.deprecated.van import VanConfig
    from transformers.src.transformers.models.deprecated.vit_hybrid import (
        ViTHybridConfig,
    )
    from transformers.src.transformers.models.deprecated.xlm_prophetnet import (
        XLMProphetNetConfig,
    )
    from transformers.src.transformers.models.depth_anything import DepthAnythingConfig
    from transformers.src.transformers.models.detr import DetrConfig
    from transformers.src.transformers.models.dinat import DinatConfig
    from transformers.src.transformers.models.dinov2 import Dinov2Config
    from transformers.src.transformers.models.distilbert import (
        DistilBertConfig,
        DistilBertTokenizer,
    )
    from transformers.src.transformers.models.donut import (
        DonutProcessor,
        DonutSwinConfig,
    )
    from transformers.src.transformers.models.dpr import (
        DPRConfig,
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )
    from transformers.src.transformers.models.dpt import DPTConfig
    from transformers.src.transformers.models.efficientnet import (
        EfficientNetConfig,
    )
    from transformers.src.transformers.models.electra import (
        ElectraConfig,
        ElectraTokenizer,
    )
    from transformers.src.transformers.models.encodec import (
        EncodecConfig,
        EncodecFeatureExtractor,
    )
    from transformers.src.transformers.models.encoder_decoder import EncoderDecoderConfig
    from transformers.src.transformers.models.ernie import ErnieConfig
    from transformers.src.transformers.models.esm import EsmConfig, EsmTokenizer
    from transformers.src.transformers.models.falcon import FalconConfig
    from transformers.src.transformers.models.falcon_mamba import FalconMambaConfig
    from transformers.src.transformers.models.fastspeech2_conformer import (
        FastSpeech2ConformerConfig,
        FastSpeech2ConformerHifiGanConfig,
        FastSpeech2ConformerTokenizer,
        FastSpeech2ConformerWithHifiGanConfig,
    )
    from transformers.src.transformers.models.flaubert import FlaubertConfig, FlaubertTokenizer
    from transformers.src.transformers.models.flava import (
        FlavaConfig,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaMultimodalConfig,
        FlavaTextConfig,
    )
    from transformers.src.transformers.models.fnet import FNetConfig
    from transformers.src.transformers.models.focalnet import FocalNetConfig
    from transformers.src.transformers.models.fsmt import (
        FSMTConfig,
        FSMTTokenizer,
    )
    from transformers.src.transformers.models.funnel import (
        FunnelConfig,
        FunnelTokenizer,
    )
    from transformers.src.transformers.models.fuyu import FuyuConfig
    from transformers.src.transformers.models.gemma import GemmaConfig
    from transformers.src.transformers.models.gemma2 import Gemma2Config
    from transformers.src.transformers.models.git import (
        GitConfig,
        GitProcessor,
        GitVisionConfig,
    )
    from transformers.src.transformers.models.glm import GlmConfig
    from transformers.src.transformers.models.glpn import GLPNConfig
    from transformers.src.transformers.models.gpt2 import (
        GPT2Config,
        GPT2Tokenizer,
    )
    from transformers.src.transformers.models.gpt_bigcode import (
        GPTBigCodeConfig,
    )
    from transformers.src.transformers.models.gpt_neo import GPTNeoConfig
    from transformers.src.transformers.models.gpt_neox import GPTNeoXConfig
    from transformers.src.transformers.models.gpt_neox_japanese import (
        GPTNeoXJapaneseConfig,
    )
    from transformers.src.transformers.models.gptj import GPTJConfig
    from transformers.src.transformers.models.granite import GraniteConfig
    from transformers.src.transformers.models.granitemoe import GraniteMoeConfig
    from transformers.src.transformers.models.grounding_dino import (
        GroundingDinoConfig,
        GroundingDinoProcessor,
    )
    from transformers.src.transformers.models.groupvit import (
        GroupViTConfig,
        GroupViTTextConfig,
        GroupViTVisionConfig,
    )
    from transformers.src.transformers.models.herbert import HerbertTokenizer
    from transformers.src.transformers.models.hiera import HieraConfig
    from transformers.src.transformers.models.hubert import HubertConfig
    from transformers.src.transformers.models.ibert import IBertConfig
    from transformers.src.transformers.models.idefics import (
        IdeficsConfig,
    )
    from transformers.src.transformers.models.idefics2 import Idefics2Config
    from transformers.src.transformers.models.idefics3 import Idefics3Config
    from transformers.src.transformers.models.imagegpt import ImageGPTConfig
    from transformers.src.transformers.models.informer import InformerConfig
    from transformers.src.transformers.models.instructblip import (
        InstructBlipConfig,
        InstructBlipProcessor,
        InstructBlipQFormerConfig,
        InstructBlipVisionConfig,
    )
    from transformers.src.transformers.models.instructblipvideo import (
        InstructBlipVideoConfig,
        InstructBlipVideoProcessor,
        InstructBlipVideoQFormerConfig,
        InstructBlipVideoVisionConfig,
    )
    from transformers.src.transformers.models.jamba import JambaConfig
    from transformers.src.transformers.models.jetmoe import JetMoeConfig
    from transformers.src.transformers.models.kosmos2 import (
        Kosmos2Config,
        Kosmos2Processor,
    )
    from transformers.src.transformers.models.layoutlm import (
        LayoutLMConfig,
        LayoutLMTokenizer,
    )
    from transformers.src.transformers.models.layoutlmv2 import (
        LayoutLMv2Config,
        LayoutLMv2FeatureExtractor,
        LayoutLMv2ImageProcessor,
        LayoutLMv2Processor,
        LayoutLMv2Tokenizer,
    )
    from transformers.src.transformers.models.layoutlmv3 import (
        LayoutLMv3Config,
        LayoutLMv3FeatureExtractor,
        LayoutLMv3ImageProcessor,
        LayoutLMv3Processor,
        LayoutLMv3Tokenizer,
    )
    from transformers.src.transformers.models.layoutxlm import LayoutXLMProcessor
    from transformers.src.transformers.models.led import LEDConfig, LEDTokenizer
    from transformers.src.transformers.models.levit import LevitConfig
    from transformers.src.transformers.models.lilt import LiltConfig
    from transformers.src.transformers.models.llama import LlamaConfig
    from transformers.src.transformers.models.llava import (
        LlavaConfig,
        LlavaProcessor,
    )
    from transformers.src.transformers.models.llava_next import (
        LlavaNextConfig,
        LlavaNextProcessor,
    )
    from transformers.src.transformers.models.llava_next_video import (
        LlavaNextVideoConfig,
        LlavaNextVideoProcessor,
    )
    from transformers.src.transformers.models.llava_onevision import (
        LlavaOnevisionConfig,
        LlavaOnevisionProcessor,
    )
    from transformers.src.transformers.models.longformer import (
        LongformerConfig,
        LongformerTokenizer,
    )
    from transformers.src.transformers.models.longt5 import LongT5Config
    from transformers.src.transformers.models.luke import (
        LukeConfig,
        LukeTokenizer,
    )
    from transformers.src.transformers.models.lxmert import (
        LxmertConfig,
        LxmertTokenizer,
    )
    from transformers.src.transformers.models.m2m_100 import M2M100Config
    from transformers.src.transformers.models.mamba import MambaConfig
    from transformers.src.transformers.models.mamba2 import Mamba2Config
    from transformers.src.transformers.models.marian import MarianConfig
    from transformers.src.transformers.models.markuplm import (
        MarkupLMConfig,
        MarkupLMFeatureExtractor,
        MarkupLMProcessor,
        MarkupLMTokenizer,
    )
    from transformers.src.transformers.models.mask2former import (
        Mask2FormerConfig,
    )
    from transformers.src.transformers.models.maskformer import (
        MaskFormerConfig,
        MaskFormerSwinConfig,
    )
    from transformers.src.transformers.models.mbart import MBartConfig
    from transformers.src.transformers.models.megatron_bert import (
        MegatronBertConfig,
    )
    from transformers.src.transformers.models.mgp_str import (
        MgpstrConfig,
        MgpstrProcessor,
        MgpstrTokenizer,
    )
    from transformers.src.transformers.models.mimi import (
        MimiConfig,
    )
    from transformers.src.transformers.models.mistral import MistralConfig
    from transformers.src.transformers.models.mixtral import MixtralConfig
    from transformers.src.transformers.models.mllama import (
        MllamaConfig,
        MllamaProcessor,
    )
    from transformers.src.transformers.models.mobilebert import (
        MobileBertConfig,
        MobileBertTokenizer,
    )
    from transformers.src.transformers.models.mobilenet_v1 import (
        MobileNetV1Config,
    )
    from transformers.src.transformers.models.mobilenet_v2 import (
        MobileNetV2Config,
    )
    from transformers.src.transformers.models.mobilevit import (
        MobileViTConfig,
    )
    from transformers.src.transformers.models.mobilevitv2 import (
        MobileViTV2Config,
    )
    from transformers.src.transformers.models.moshi import (
        MoshiConfig,
        MoshiDepthConfig,
    )
    from transformers.src.transformers.models.mpnet import (
        MPNetConfig,
        MPNetTokenizer,
    )
    from transformers.src.transformers.models.mpt import MptConfig
    from transformers.src.transformers.models.mra import MraConfig
    from transformers.src.transformers.models.mt5 import MT5Config
    from transformers.src.transformers.models.musicgen import (
        MusicgenConfig,
        MusicgenDecoderConfig,
    )
    from transformers.src.transformers.models.musicgen_melody import (
        MusicgenMelodyConfig,
        MusicgenMelodyDecoderConfig,
    )
    from transformers.src.transformers.models.mvp import MvpConfig, MvpTokenizer
    from transformers.src.transformers.models.myt5 import MyT5Tokenizer
    from transformers.src.transformers.models.nemotron import NemotronConfig
    from transformers.src.transformers.models.nllb_moe import NllbMoeConfig
    from transformers.src.transformers.models.nougat import NougatProcessor
    from transformers.src.transformers.models.nystromformer import (
        NystromformerConfig,
    )
    from transformers.src.transformers.models.olmo import OlmoConfig
    from transformers.src.transformers.models.olmoe import OlmoeConfig
    from transformers.src.transformers.models.omdet_turbo import (
        OmDetTurboConfig,
        OmDetTurboProcessor,
    )
    from transformers.src.transformers.models.oneformer import (
        OneFormerConfig,
        OneFormerProcessor,
    )
    from transformers.src.transformers.models.openai import (
        OpenAIGPTConfig,
        OpenAIGPTTokenizer,
    )
    from transformers.src.transformers.models.opt import OPTConfig
    from transformers.src.transformers.models.owlv2 import (
        Owlv2Config,
        Owlv2Processor,
        Owlv2TextConfig,
        Owlv2VisionConfig,
    )
    from transformers.src.transformers.models.owlvit import (
        OwlViTConfig,
        OwlViTProcessor,
        OwlViTTextConfig,
        OwlViTVisionConfig,
    )
    from transformers.src.transformers.models.paligemma import (
        PaliGemmaConfig,
    )
    from transformers.src.transformers.models.patchtsmixer import (
        PatchTSMixerConfig,
    )
    from transformers.src.transformers.models.patchtst import PatchTSTConfig
    from transformers.src.transformers.models.pegasus import (
        PegasusConfig,
        PegasusTokenizer,
    )
    from transformers.src.transformers.models.pegasus_x import (
        PegasusXConfig,
    )
    from transformers.src.transformers.models.perceiver import (
        PerceiverConfig,
        PerceiverTokenizer,
    )
    from transformers.src.transformers.models.persimmon import (
        PersimmonConfig,
    )
    from transformers.src.transformers.models.phi import PhiConfig
    from transformers.src.transformers.models.phi3 import Phi3Config
    from transformers.src.transformers.models.phimoe import PhimoeConfig
    from transformers.src.transformers.models.phobert import PhobertTokenizer
    from transformers.src.transformers.models.pix2struct import (
        Pix2StructConfig,
        Pix2StructProcessor,
        Pix2StructTextConfig,
        Pix2StructVisionConfig,
    )
    from transformers.src.transformers.models.pixtral import (
        PixtralProcessor,
        PixtralVisionConfig,
    )
    from transformers.src.transformers.models.plbart import PLBartConfig
    from transformers.src.transformers.models.poolformer import (
        PoolFormerConfig,
    )
    from transformers.src.transformers.models.pop2piano import (
        Pop2PianoConfig,
    )
    from transformers.src.transformers.models.prophetnet import (
        ProphetNetConfig,
        ProphetNetTokenizer,
    )
    from transformers.src.transformers.models.pvt import PvtConfig
    from transformers.src.transformers.models.pvt_v2 import PvtV2Config
    from transformers.src.transformers.models.qwen2 import Qwen2Config, Qwen2Tokenizer
    from transformers.src.transformers.models.qwen2_audio import (
        Qwen2AudioConfig,
        Qwen2AudioEncoderConfig,
        Qwen2AudioProcessor,
    )
    from transformers.src.transformers.models.qwen2_moe import Qwen2MoeConfig
    from transformers.src.transformers.models.qwen2_vl import (
        Qwen2VLConfig,
        Qwen2VLProcessor,
    )
    from transformers.src.transformers.models.rag import RagConfig, RagRetriever, RagTokenizer
    from transformers.src.transformers.models.recurrent_gemma import RecurrentGemmaConfig
    from transformers.src.transformers.models.reformer import ReformerConfig
    from transformers.src.transformers.models.regnet import RegNetConfig
    from transformers.src.transformers.models.rembert import RemBertConfig
    from transformers.src.transformers.models.resnet import ResNetConfig
    from transformers.src.transformers.models.roberta import (
        RobertaConfig,
        RobertaTokenizer,
    )
    from transformers.src.transformers.models.roberta_prelayernorm import (
        RobertaPreLayerNormConfig,
    )
    from transformers.src.transformers.models.roc_bert import (
        RoCBertConfig,
        RoCBertTokenizer,
    )
    from transformers.src.transformers.models.roformer import (
        RoFormerConfig,
        RoFormerTokenizer,
    )
    from transformers.src.transformers.models.rt_detr import (
        RTDetrConfig,
        RTDetrResNetConfig,
    )
    from transformers.src.transformers.models.rwkv import RwkvConfig
    from transformers.src.transformers.models.sam import (
        SamConfig,
        SamMaskDecoderConfig,
        SamProcessor,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )
    from transformers.src.transformers.models.seamless_m4t import (
        SeamlessM4TConfig,
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
    )
    from transformers.src.transformers.models.seamless_m4t_v2 import (
        SeamlessM4Tv2Config,
    )
    from transformers.src.transformers.models.segformer import SegformerConfig
    from transformers.src.transformers.models.seggpt import SegGptConfig
    from transformers.src.transformers.models.sew import SEWConfig
    from transformers.src.transformers.models.sew_d import SEWDConfig
    from transformers.src.transformers.models.siglip import (
        SiglipConfig,
        SiglipProcessor,
        SiglipTextConfig,
        SiglipVisionConfig,
    )
    from transformers.src.transformers.models.speech_encoder_decoder import SpeechEncoderDecoderConfig
    from transformers.src.transformers.models.speech_to_text import (
        Speech2TextConfig,
        Speech2TextFeatureExtractor,
        Speech2TextProcessor,
    )
    from transformers.src.transformers.models.speecht5 import (
        SpeechT5Config,
        SpeechT5FeatureExtractor,
        SpeechT5HifiGanConfig,
        SpeechT5Processor,
    )
    from transformers.src.transformers.models.splinter import (
        SplinterConfig,
        SplinterTokenizer,
    )
    from transformers.src.transformers.models.squeezebert import (
        SqueezeBertConfig,
        SqueezeBertTokenizer,
    )
    from transformers.src.transformers.models.stablelm import StableLmConfig
    from transformers.src.transformers.models.starcoder2 import Starcoder2Config
    from transformers.src.transformers.models.superpoint import SuperPointConfig
    from transformers.src.transformers.models.swiftformer import (
        SwiftFormerConfig,
    )
    from transformers.src.transformers.models.swin import SwinConfig
    from transformers.src.transformers.models.swin2sr import Swin2SRConfig
    from transformers.src.transformers.models.swinv2 import Swinv2Config
    from transformers.src.transformers.models.switch_transformers import (
        SwitchTransformersConfig,
    )
    from transformers.src.transformers.models.t5 import T5Config
    from transformers.src.transformers.models.table_transformer import (
        TableTransformerConfig,
    )
    from transformers.src.transformers.models.tapas import (
        TapasConfig,
        TapasTokenizer,
    )
    from transformers.src.transformers.models.time_series_transformer import (
        TimeSeriesTransformerConfig,
    )
    from transformers.src.transformers.models.timesformer import (
        TimesformerConfig,
    )
    from transformers.src.transformers.models.timm_backbone import TimmBackboneConfig
    from transformers.src.transformers.models.trocr import (
        TrOCRConfig,
        TrOCRProcessor,
    )
    from transformers.src.transformers.models.tvp import (
        TvpConfig,
        TvpProcessor,
    )
    from transformers.src.transformers.models.udop import UdopConfig, UdopProcessor
    from transformers.src.transformers.models.umt5 import UMT5Config
    from transformers.src.transformers.models.unispeech import (
        UniSpeechConfig,
    )
    from transformers.src.transformers.models.unispeech_sat import (
        UniSpeechSatConfig,
    )
    from transformers.src.transformers.models.univnet import (
        UnivNetConfig,
        UnivNetFeatureExtractor,
    )
    from transformers.src.transformers.models.upernet import UperNetConfig
    from transformers.src.transformers.models.video_llava import VideoLlavaConfig
    from transformers.src.transformers.models.videomae import VideoMAEConfig
    from transformers.src.transformers.models.vilt import (
        ViltConfig,
        ViltFeatureExtractor,
        ViltImageProcessor,
        ViltProcessor,
    )
    from transformers.src.transformers.models.vipllava import (
        VipLlavaConfig,
    )
    from transformers.src.transformers.models.vision_encoder_decoder import VisionEncoderDecoderConfig
    from transformers.src.transformers.models.vision_text_dual_encoder import (
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderProcessor,
    )
    from transformers.src.transformers.models.visual_bert import (
        VisualBertConfig,
    )
    from transformers.src.transformers.models.vit import ViTConfig
    from transformers.src.transformers.models.vit_mae import ViTMAEConfig
    from transformers.src.transformers.models.vit_msn import ViTMSNConfig
    from transformers.src.transformers.models.vitdet import VitDetConfig
    from transformers.src.transformers.models.vitmatte import VitMatteConfig
    from transformers.src.transformers.models.vits import (
        VitsConfig,
        VitsTokenizer,
    )
    from transformers.src.transformers.models.vivit import VivitConfig
    from transformers.src.transformers.models.wav2vec2 import (
        Wav2Vec2Config,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        Wav2Vec2Tokenizer,
    )
    from transformers.src.transformers.models.wav2vec2_bert import (
        Wav2Vec2BertConfig,
        Wav2Vec2BertProcessor,
    )
    from transformers.src.transformers.models.wav2vec2_conformer import (
        Wav2Vec2ConformerConfig,
    )
    from transformers.src.transformers.models.wav2vec2_phoneme import Wav2Vec2PhonemeCTCTokenizer
    from transformers.src.transformers.models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM
    from transformers.src.transformers.models.wavlm import WavLMConfig
    from transformers.src.transformers.models.whisper import (
        WhisperConfig,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )
    from transformers.src.transformers.models.x_clip import (
        XCLIPConfig,
        XCLIPProcessor,
        XCLIPTextConfig,
        XCLIPVisionConfig,
    )
    from transformers.src.transformers.models.xglm import XGLMConfig
    from transformers.src.transformers.models.xlm import XLMConfig, XLMTokenizer
    from transformers.src.transformers.models.xlm_roberta import (
        XLMRobertaConfig,
    )
    from transformers.src.transformers.models.xlm_roberta_xl import (
        XLMRobertaXLConfig,
    )
    from transformers.src.transformers.models.xlnet import XLNetConfig
    from transformers.src.transformers.models.xmod import XmodConfig
    from transformers.src.transformers.models.yolos import YolosConfig
    from transformers.src.transformers.models.yoso import YosoConfig
    from transformers.src.transformers.models.zamba import ZambaConfig
    from transformers.src.transformers.models.zoedepth import ZoeDepthConfig

    # Pipelines
    from transformers.src.transformers.pipelines import (
        AudioClassificationPipeline,
        AutomaticSpeechRecognitionPipeline,
        CsvPipelineDataFormat,
        DepthEstimationPipeline,
        DocumentQuestionAnsweringPipeline,
        FeatureExtractionPipeline,
        FillMaskPipeline,
        ImageClassificationPipeline,
        ImageFeatureExtractionPipeline,
        ImageSegmentationPipeline,
        ImageTextToTextPipeline,
        ImageToImagePipeline,
        ImageToTextPipeline,
        JsonPipelineDataFormat,
        MaskGenerationPipeline,
        NerPipeline,
        ObjectDetectionPipeline,
        PipedPipelineDataFormat,
        Pipeline,
        PipelineDataFormat,
        QuestionAnsweringPipeline,
        SummarizationPipeline,
        TableQuestionAnsweringPipeline,
        Text2TextGenerationPipeline,
        TextClassificationPipeline,
        TextGenerationPipeline,
        TextToAudioPipeline,
        TokenClassificationPipeline,
        TranslationPipeline,
        VideoClassificationPipeline,
        VisualQuestionAnsweringPipeline,
        ZeroShotAudioClassificationPipeline,
        ZeroShotClassificationPipeline,
        ZeroShotImageClassificationPipeline,
        ZeroShotObjectDetectionPipeline,
        pipeline,
    )
    from transformers.src.transformers.processing_utils import ProcessorMixin

    # Tokenization
    from transformers.src.transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.src.transformers.tokenization_utils_base import (
        AddedToken,
        BatchEncoding,
        CharSpan,
        PreTrainedTokenizerBase,
        SpecialTokensMixin,
        TokenSpan,
    )

    # Trainer
    from transformers.src.transformers.trainer_callback import (
        DefaultFlowCallback,
        EarlyStoppingCallback,
        PrinterCallback,
        ProgressCallback,
        TrainerCallback,
        TrainerControl,
        TrainerState,
    )
    from transformers.src.transformers.trainer_utils import (
        EvalPrediction,
        IntervalStrategy,
        SchedulerType,
        enable_full_determinism,
        set_seed,
    )
    from transformers.src.transformers.training_args import TrainingArguments
    from transformers.src.transformers.training_args_seq2seq import Seq2SeqTrainingArguments
    from transformers.src.transformers.training_args_tf import TFTrainingArguments

    # Files and general utilities
    from transformers.src.transformers.utils import (
        CONFIG_NAME,
        MODEL_CARD_NAME,
        PYTORCH_PRETRAINED_BERT_CACHE,
        PYTORCH_TRANSFORMERS_CACHE,
        SPIECE_UNDERLINE,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        TRANSFORMERS_CACHE,
        WEIGHTS_NAME,
        TensorType,
        add_end_docstrings,
        add_start_docstrings,
        is_apex_available,
        is_av_available,
        is_bitsandbytes_available,
        is_datasets_available,
        is_faiss_available,
        is_flax_available,
        is_keras_nlp_available,
        is_phonemizer_available,
        is_psutil_available,
        is_py3nvml_available,
        is_pyctcdecode_available,
        is_sacremoses_available,
        is_safetensors_available,
        is_scipy_available,
        is_sentencepiece_available,
        is_sklearn_available,
        is_speech_available,
        is_tensorflow_text_available,
        is_tf_available,
        is_timm_available,
        is_tokenizers_available,
        is_torch_available,
        is_torch_mlu_available,
        is_torch_musa_available,
        is_torch_neuroncore_available,
        is_torch_npu_available,
        is_torch_tpu_available,
        is_torch_xla_available,
        is_torch_xpu_available,
        is_torchvision_available,
        is_vision_available,
        logging,
    )

    # bitsandbytes config
    from transformers.src.transformers.utils.quantization_config import (
        AqlmConfig,
        AwqConfig,
        BitNetConfig,
        BitsAndBytesConfig,
        CompressedTensorsConfig,
        EetqConfig,
        FbgemmFp8Config,
        GPTQConfig,
        HqqConfig,
        QuantoConfig,
        TorchAoConfig,
    )

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_sentencepiece_objects import *
    else:
        from transformers.src.transformers.models.albert import AlbertTokenizer
        from transformers.src.transformers.models.barthez import BarthezTokenizer
        from transformers.src.transformers.models.bartpho import BartphoTokenizer
        from transformers.src.transformers.models.bert_generation import BertGenerationTokenizer
        from transformers.src.transformers.models.big_bird import BigBirdTokenizer
        from transformers.src.transformers.models.camembert import CamembertTokenizer
        from transformers.src.transformers.models.code_llama import CodeLlamaTokenizer
        from transformers.src.transformers.models.cpm import CpmTokenizer
        from transformers.src.transformers.models.deberta_v2 import DebertaV2Tokenizer
        from transformers.src.transformers.models.deprecated.ernie_m import ErnieMTokenizer
        from transformers.src.transformers.models.deprecated.xlm_prophetnet import XLMProphetNetTokenizer
        from transformers.src.transformers.models.fnet import FNetTokenizer
        from transformers.src.transformers.models.gemma import GemmaTokenizer
        from transformers.src.transformers.models.gpt_sw3 import GPTSw3Tokenizer
        from transformers.src.transformers.models.layoutxlm import LayoutXLMTokenizer
        from transformers.src.transformers.models.llama import LlamaTokenizer
        from transformers.src.transformers.models.m2m_100 import M2M100Tokenizer
        from transformers.src.transformers.models.marian import MarianTokenizer
        from transformers.src.transformers.models.mbart import MBartTokenizer
        from transformers.src.transformers.models.mbart50 import MBart50Tokenizer
        from transformers.src.transformers.models.mluke import MLukeTokenizer
        from transformers.src.transformers.models.mt5 import MT5Tokenizer
        from transformers.src.transformers.models.nllb import NllbTokenizer
        from transformers.src.transformers.models.pegasus import PegasusTokenizer
        from transformers.src.transformers.models.plbart import PLBartTokenizer
        from transformers.src.transformers.models.reformer import ReformerTokenizer
        from transformers.src.transformers.models.rembert import RemBertTokenizer
        from transformers.src.transformers.models.seamless_m4t import SeamlessM4TTokenizer
        from transformers.src.transformers.models.siglip import SiglipTokenizer
        from transformers.src.transformers.models.speech_to_text import Speech2TextTokenizer
        from transformers.src.transformers.models.speecht5 import SpeechT5Tokenizer
        from transformers.src.transformers.models.t5 import T5Tokenizer
        from transformers.src.transformers.models.udop import UdopTokenizer
        from transformers.src.transformers.models.xglm import XGLMTokenizer
        from transformers.src.transformers.models.xlm_roberta import XLMRobertaTokenizer
        from transformers.src.transformers.models.xlnet import XLNetTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_tokenizers_objects import *
    else:
        # Fast tokenizers imports
        from transformers.src.transformers.models.albert import AlbertTokenizerFast
        from transformers.src.transformers.models.bart import BartTokenizerFast
        from transformers.src.transformers.models.barthez import BarthezTokenizerFast
        from transformers.src.transformers.models.bert import BertTokenizerFast
        from transformers.src.transformers.models.big_bird import BigBirdTokenizerFast
        from transformers.src.transformers.models.blenderbot import BlenderbotTokenizerFast
        from transformers.src.transformers.models.blenderbot_small import BlenderbotSmallTokenizerFast
        from transformers.src.transformers.models.bloom import BloomTokenizerFast
        from transformers.src.transformers.models.camembert import CamembertTokenizerFast
        from transformers.src.transformers.models.clip import CLIPTokenizerFast
        from transformers.src.transformers.models.code_llama import CodeLlamaTokenizerFast
        from transformers.src.transformers.models.codegen import CodeGenTokenizerFast
        from transformers.src.transformers.models.cohere import CohereTokenizerFast
        from transformers.src.transformers.models.convbert import ConvBertTokenizerFast
        from transformers.src.transformers.models.cpm import CpmTokenizerFast
        from transformers.src.transformers.models.deberta import DebertaTokenizerFast
        from transformers.src.transformers.models.deberta_v2 import DebertaV2TokenizerFast
        from transformers.src.transformers.models.deprecated.realm import RealmTokenizerFast
        from transformers.src.transformers.models.deprecated.retribert import RetriBertTokenizerFast
        from transformers.src.transformers.models.distilbert import DistilBertTokenizerFast
        from transformers.src.transformers.models.dpr import (
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoderTokenizerFast,
            DPRReaderTokenizerFast,
        )
        from transformers.src.transformers.models.electra import ElectraTokenizerFast
        from transformers.src.transformers.models.fnet import FNetTokenizerFast
        from transformers.src.transformers.models.funnel import FunnelTokenizerFast
        from transformers.src.transformers.models.gemma import GemmaTokenizerFast
        from transformers.src.transformers.models.gpt2 import GPT2TokenizerFast
        from transformers.src.transformers.models.gpt_neox import GPTNeoXTokenizerFast
        from transformers.src.transformers.models.gpt_neox_japanese import GPTNeoXJapaneseTokenizer
        from transformers.src.transformers.models.herbert import HerbertTokenizerFast
        from transformers.src.transformers.models.layoutlm import LayoutLMTokenizerFast
        from transformers.src.transformers.models.layoutlmv2 import LayoutLMv2TokenizerFast
        from transformers.src.transformers.models.layoutlmv3 import LayoutLMv3TokenizerFast
        from transformers.src.transformers.models.layoutxlm import LayoutXLMTokenizerFast
        from transformers.src.transformers.models.led import LEDTokenizerFast
        from transformers.src.transformers.models.llama import LlamaTokenizerFast
        from transformers.src.transformers.models.longformer import LongformerTokenizerFast
        from transformers.src.transformers.models.lxmert import LxmertTokenizerFast
        from transformers.src.transformers.models.markuplm import MarkupLMTokenizerFast
        from transformers.src.transformers.models.mbart import MBartTokenizerFast
        from transformers.src.transformers.models.mbart50 import MBart50TokenizerFast
        from transformers.src.transformers.models.mobilebert import MobileBertTokenizerFast
        from transformers.src.transformers.models.mpnet import MPNetTokenizerFast
        from transformers.src.transformers.models.mt5 import MT5TokenizerFast
        from transformers.src.transformers.models.mvp import MvpTokenizerFast
        from transformers.src.transformers.models.nllb import NllbTokenizerFast
        from transformers.src.transformers.models.nougat import NougatTokenizerFast
        from transformers.src.transformers.models.openai import OpenAIGPTTokenizerFast
        from transformers.src.transformers.models.pegasus import PegasusTokenizerFast
        from transformers.src.transformers.models.qwen2 import Qwen2TokenizerFast
        from transformers.src.transformers.models.reformer import ReformerTokenizerFast
        from transformers.src.transformers.models.rembert import RemBertTokenizerFast
        from transformers.src.transformers.models.roberta import RobertaTokenizerFast
        from transformers.src.transformers.models.roformer import RoFormerTokenizerFast
        from transformers.src.transformers.models.seamless_m4t import SeamlessM4TTokenizerFast
        from transformers.src.transformers.models.splinter import SplinterTokenizerFast
        from transformers.src.transformers.models.squeezebert import SqueezeBertTokenizerFast
        from transformers.src.transformers.models.t5 import T5TokenizerFast
        from transformers.src.transformers.models.udop import UdopTokenizerFast
        from transformers.src.transformers.models.whisper import WhisperTokenizerFast
        from transformers.src.transformers.models.xglm import XGLMTokenizerFast
        from transformers.src.transformers.models.xlm_roberta import XLMRobertaTokenizerFast
        from transformers.src.transformers.models.xlnet import XLNetTokenizerFast
        from transformers.src.transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    try:
        if not (is_sentencepiece_available() and is_tokenizers_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummies_sentencepiece_and_tokenizers_objects import *
    else:
        from transformers.src.transformers.convert_slow_tokenizer import (
            SLOW_TO_FAST_CONVERTERS,
            convert_slow_tokenizer,
        )

    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_tensorflow_text_objects import *
    else:
        from transformers.src.transformers.models.bert import TFBertTokenizer

    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_keras_nlp_objects import *
    else:
        from transformers.src.transformers.models.gpt2 import TFGPT2Tokenizer

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_vision_objects import *
    else:
        from transformers.src.transformers.image_processing_base import ImageProcessingMixin
        from transformers.src.transformers.image_processing_utils import BaseImageProcessor
        from transformers.src.transformers.image_utils import ImageFeatureExtractionMixin
        from transformers.src.transformers.models.beit import BeitFeatureExtractor, BeitImageProcessor
        from transformers.src.transformers.models.bit import BitImageProcessor
        from transformers.src.transformers.models.blip import BlipImageProcessor
        from transformers.src.transformers.models.bridgetower import BridgeTowerImageProcessor
        from transformers.src.transformers.models.chameleon import ChameleonImageProcessor
        from transformers.src.transformers.models.chinese_clip import (
            ChineseCLIPFeatureExtractor,
            ChineseCLIPImageProcessor,
        )
        from transformers.src.transformers.models.clip import CLIPFeatureExtractor, CLIPImageProcessor
        from transformers.src.transformers.models.conditional_detr import (
            ConditionalDetrFeatureExtractor,
            ConditionalDetrImageProcessor,
        )
        from transformers.src.transformers.models.convnext import ConvNextFeatureExtractor, ConvNextImageProcessor
        from transformers.src.transformers.models.deformable_detr import (
            DeformableDetrFeatureExtractor,
            DeformableDetrImageProcessor,
        )
        from transformers.src.transformers.models.deit import DeiTFeatureExtractor, DeiTImageProcessor
        from transformers.src.transformers.models.deprecated.deta import DetaImageProcessor
        from transformers.src.transformers.models.deprecated.efficientformer import EfficientFormerImageProcessor
        from transformers.src.transformers.models.deprecated.tvlt import TvltImageProcessor
        from transformers.src.transformers.models.deprecated.vit_hybrid import ViTHybridImageProcessor
        from transformers.src.transformers.models.detr import DetrFeatureExtractor, DetrImageProcessor, DetrImageProcessorFast
        from transformers.src.transformers.models.donut import DonutFeatureExtractor, DonutImageProcessor
        from transformers.src.transformers.models.dpt import DPTFeatureExtractor, DPTImageProcessor
        from transformers.src.transformers.models.efficientnet import EfficientNetImageProcessor
        from transformers.src.transformers.models.flava import (
            FlavaFeatureExtractor,
            FlavaImageProcessor,
            FlavaProcessor,
        )
        from transformers.src.transformers.models.fuyu import FuyuImageProcessor, FuyuProcessor
        from transformers.src.transformers.models.glpn import GLPNFeatureExtractor, GLPNImageProcessor
        from transformers.src.transformers.models.grounding_dino import GroundingDinoImageProcessor
        from transformers.src.transformers.models.idefics import IdeficsImageProcessor
        from transformers.src.transformers.models.idefics2 import Idefics2ImageProcessor
        from transformers.src.transformers.models.idefics3 import Idefics3ImageProcessor
        from transformers.src.transformers.models.imagegpt import ImageGPTFeatureExtractor, ImageGPTImageProcessor
        from transformers.src.transformers.models.instructblipvideo import InstructBlipVideoImageProcessor
        from transformers.src.transformers.models.layoutlmv2 import (
            LayoutLMv2FeatureExtractor,
            LayoutLMv2ImageProcessor,
        )
        from transformers.src.transformers.models.layoutlmv3 import (
            LayoutLMv3FeatureExtractor,
            LayoutLMv3ImageProcessor,
        )
        from transformers.src.transformers.models.levit import LevitFeatureExtractor, LevitImageProcessor
        from transformers.src.transformers.models.llava_next import LlavaNextImageProcessor
        from transformers.src.transformers.models.llava_next_video import LlavaNextVideoImageProcessor
        from transformers.src.transformers.models.llava_onevision import LlavaOnevisionImageProcessor, LlavaOnevisionVideoProcessor
        from transformers.src.transformers.models.mask2former import Mask2FormerImageProcessor
        from transformers.src.transformers.models.maskformer import (
            MaskFormerFeatureExtractor,
            MaskFormerImageProcessor,
        )
        from transformers.src.transformers.models.mllama import MllamaImageProcessor
        from transformers.src.transformers.models.mobilenet_v1 import (
            MobileNetV1FeatureExtractor,
            MobileNetV1ImageProcessor,
        )
        from transformers.src.transformers.models.mobilenet_v2 import (
            MobileNetV2FeatureExtractor,
            MobileNetV2ImageProcessor,
        )
        from transformers.src.transformers.models.mobilevit import MobileViTFeatureExtractor, MobileViTImageProcessor
        from transformers.src.transformers.models.nougat import NougatImageProcessor
        from transformers.src.transformers.models.oneformer import OneFormerImageProcessor
        from transformers.src.transformers.models.owlv2 import Owlv2ImageProcessor
        from transformers.src.transformers.models.owlvit import OwlViTFeatureExtractor, OwlViTImageProcessor
        from transformers.src.transformers.models.perceiver import PerceiverFeatureExtractor, PerceiverImageProcessor
        from transformers.src.transformers.models.pix2struct import Pix2StructImageProcessor
        from transformers.src.transformers.models.pixtral import PixtralImageProcessor
        from transformers.src.transformers.models.poolformer import (
            PoolFormerFeatureExtractor,
            PoolFormerImageProcessor,
        )
        from transformers.src.transformers.models.pvt import PvtImageProcessor
        from transformers.src.transformers.models.qwen2_vl import Qwen2VLImageProcessor
        from transformers.src.transformers.models.rt_detr import RTDetrImageProcessor, RTDetrImageProcessorFast
        from transformers.src.transformers.models.sam import SamImageProcessor
        from transformers.src.transformers.models.segformer import SegformerFeatureExtractor, SegformerImageProcessor
        from transformers.src.transformers.models.seggpt import SegGptImageProcessor
        from transformers.src.transformers.models.siglip import SiglipImageProcessor
        from transformers.src.transformers.models.superpoint import SuperPointImageProcessor
        from transformers.src.transformers.models.swin2sr import Swin2SRImageProcessor
        from transformers.src.transformers.models.tvp import TvpImageProcessor
        from transformers.src.transformers.models.video_llava import VideoLlavaImageProcessor
        from transformers.src.transformers.models.videomae import VideoMAEFeatureExtractor, VideoMAEImageProcessor
        from transformers.src.transformers.models.vilt import ViltFeatureExtractor, ViltImageProcessor, ViltProcessor
        from transformers.src.transformers.models.vit import ViTFeatureExtractor, ViTImageProcessor
        from transformers.src.transformers.models.vitmatte import VitMatteImageProcessor
        from transformers.src.transformers.models.vivit import VivitImageProcessor
        from transformers.src.transformers.models.yolos import YolosFeatureExtractor, YolosImageProcessor
        from transformers.src.transformers.models.zoedepth import ZoeDepthImageProcessor

    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_torchvision_objects import *
    else:
        from transformers.src.transformers.image_processing_utils_fast import BaseImageProcessorFast
        from transformers.src.transformers.models.vit import ViTImageProcessorFast

    # Modeling
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_pt_objects import *
    else:
        # Benchmarks
        from transformers.src.transformers.benchmark.benchmark import PyTorchBenchmark
        from transformers.src.transformers.benchmark.benchmark_args import PyTorchBenchmarkArguments
        from transformers.src.transformers.cache_utils import (
            Cache,
            CacheConfig,
            DynamicCache,
            EncoderDecoderCache,
            HQQQuantizedCache,
            HybridCache,
            MambaCache,
            OffloadedCache,
            OffloadedStaticCache,
            QuantizedCache,
            QuantizedCacheConfig,
            QuantoQuantizedCache,
            SinkCache,
            SlidingWindowCache,
            StaticCache,
        )
        from transformers.src.transformers.data.datasets import (
            GlueDataset,
            GlueDataTrainingArguments,
            LineByLineTextDataset,
            LineByLineWithRefDataset,
            LineByLineWithSOPTextDataset,
            SquadDataset,
            SquadDataTrainingArguments,
            TextDataset,
            TextDatasetForNextSentencePrediction,
        )
        from transformers.src.transformers.generation import (
            AlternatingCodebooksLogitsProcessor,
            BayesianDetectorConfig,
            BayesianDetectorModel,
            BeamScorer,
            BeamSearchScorer,
            ClassifierFreeGuidanceLogitsProcessor,
            ConstrainedBeamSearchScorer,
            Constraint,
            ConstraintListState,
            DisjunctiveConstraint,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EosTokenCriteria,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            GenerationMixin,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            LogitsWarper,
            MaxLengthCriteria,
            MaxTimeCriteria,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PhrasalConstraint,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            SynthIDTextWatermarkDetector,
            SynthIDTextWatermarkingConfig,
            SynthIDTextWatermarkLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkDetector,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from transformers.src.transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )
        from transformers.src.transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from transformers.src.transformers.modeling_utils import PreTrainedModel
        from transformers.src.transformers.models.albert import (
            AlbertForMaskedLM,
            AlbertForMultipleChoice,
            AlbertForPreTraining,
            AlbertForQuestionAnswering,
            AlbertForSequenceClassification,
            AlbertForTokenClassification,
            AlbertModel,
            AlbertPreTrainedModel,
            load_tf_weights_in_albert,
        )
        from transformers.src.transformers.models.align import (
            AlignModel,
            AlignPreTrainedModel,
            AlignTextModel,
            AlignVisionModel,
        )
        from transformers.src.transformers.models.altclip import (
            AltCLIPModel,
            AltCLIPPreTrainedModel,
            AltCLIPTextModel,
            AltCLIPVisionModel,
        )
        from transformers.src.transformers.models.audio_spectrogram_transformer import (
            ASTForAudioClassification,
            ASTModel,
            ASTPreTrainedModel,
        )
        from transformers.src.transformers.models.auto import (
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
            MODEL_FOR_AUDIO_XVECTOR_MAPPING,
            MODEL_FOR_BACKBONE_MAPPING,
            MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
            MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_IMAGE_MAPPING,
            MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
            MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
            MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
            MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
            MODEL_FOR_MASK_GENERATION_MAPPING,
            MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_OBJECT_DETECTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TEXT_ENCODING_MAPPING,
            MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
            MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
            MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
            MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
            MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
            MODEL_FOR_VISION_2_SEQ_MAPPING,
            MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoBackbone,
            AutoModel,
            AutoModelForAudioClassification,
            AutoModelForAudioFrameClassification,
            AutoModelForAudioXVector,
            AutoModelForCausalLM,
            AutoModelForCTC,
            AutoModelForDepthEstimation,
            AutoModelForDocumentQuestionAnswering,
            AutoModelForImageClassification,
            AutoModelForImageSegmentation,
            AutoModelForImageTextToText,
            AutoModelForImageToImage,
            AutoModelForInstanceSegmentation,
            AutoModelForKeypointDetection,
            AutoModelForMaskedImageModeling,
            AutoModelForMaskedLM,
            AutoModelForMaskGeneration,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForObjectDetection,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSemanticSegmentation,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForSpeechSeq2Seq,
            AutoModelForTableQuestionAnswering,
            AutoModelForTextEncoding,
            AutoModelForTextToSpectrogram,
            AutoModelForTextToWaveform,
            AutoModelForTokenClassification,
            AutoModelForUniversalSegmentation,
            AutoModelForVideoClassification,
            AutoModelForVision2Seq,
            AutoModelForVisualQuestionAnswering,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
            AutoModelWithLMHead,
        )
        from transformers.src.transformers.models.autoformer import (
            AutoformerForPrediction,
            AutoformerModel,
            AutoformerPreTrainedModel,
        )
        from transformers.src.transformers.models.bark import (
            BarkCausalModel,
            BarkCoarseModel,
            BarkFineModel,
            BarkModel,
            BarkPreTrainedModel,
            BarkSemanticModel,
        )
        from transformers.src.transformers.models.bart import (
            BartForCausalLM,
            BartForConditionalGeneration,
            BartForQuestionAnswering,
            BartForSequenceClassification,
            BartModel,
            BartPreTrainedModel,
            BartPretrainedModel,
            PretrainedBartModel,
        )
        from transformers.src.transformers.models.beit import (
            BeitBackbone,
            BeitForImageClassification,
            BeitForMaskedImageModeling,
            BeitForSemanticSegmentation,
            BeitModel,
            BeitPreTrainedModel,
        )
        from transformers.src.transformers.models.bert import (
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            load_tf_weights_in_bert,
        )
        from transformers.src.transformers.models.bert_generation import (
            BertGenerationDecoder,
            BertGenerationEncoder,
            BertGenerationPreTrainedModel,
            load_tf_weights_in_bert_generation,
        )
        from transformers.src.transformers.models.big_bird import (
            BigBirdForCausalLM,
            BigBirdForMaskedLM,
            BigBirdForMultipleChoice,
            BigBirdForPreTraining,
            BigBirdForQuestionAnswering,
            BigBirdForSequenceClassification,
            BigBirdForTokenClassification,
            BigBirdModel,
            BigBirdPreTrainedModel,
            load_tf_weights_in_big_bird,
        )
        from transformers.src.transformers.models.bigbird_pegasus import (
            BigBirdPegasusForCausalLM,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForQuestionAnswering,
            BigBirdPegasusForSequenceClassification,
            BigBirdPegasusModel,
            BigBirdPegasusPreTrainedModel,
        )
        from transformers.src.transformers.models.biogpt import (
            BioGptForCausalLM,
            BioGptForSequenceClassification,
            BioGptForTokenClassification,
            BioGptModel,
            BioGptPreTrainedModel,
        )
        from transformers.src.transformers.models.bit import (
            BitBackbone,
            BitForImageClassification,
            BitModel,
            BitPreTrainedModel,
        )
        from transformers.src.transformers.models.blenderbot import (
            BlenderbotForCausalLM,
            BlenderbotForConditionalGeneration,
            BlenderbotModel,
            BlenderbotPreTrainedModel,
        )
        from transformers.src.transformers.models.blenderbot_small import (
            BlenderbotSmallForCausalLM,
            BlenderbotSmallForConditionalGeneration,
            BlenderbotSmallModel,
            BlenderbotSmallPreTrainedModel,
        )
        from transformers.src.transformers.models.blip import (
            BlipForConditionalGeneration,
            BlipForImageTextRetrieval,
            BlipForQuestionAnswering,
            BlipModel,
            BlipPreTrainedModel,
            BlipTextModel,
            BlipVisionModel,
        )
        from transformers.src.transformers.models.blip_2 import (
            Blip2ForConditionalGeneration,
            Blip2ForImageTextRetrieval,
            Blip2Model,
            Blip2PreTrainedModel,
            Blip2QFormerModel,
            Blip2TextModelWithProjection,
            Blip2VisionModel,
            Blip2VisionModelWithProjection,
        )
        from transformers.src.transformers.models.bloom import (
            BloomForCausalLM,
            BloomForQuestionAnswering,
            BloomForSequenceClassification,
            BloomForTokenClassification,
            BloomModel,
            BloomPreTrainedModel,
        )
        from transformers.src.transformers.models.bridgetower import (
            BridgeTowerForContrastiveLearning,
            BridgeTowerForImageAndTextRetrieval,
            BridgeTowerForMaskedLM,
            BridgeTowerModel,
            BridgeTowerPreTrainedModel,
        )
        from transformers.src.transformers.models.bros import (
            BrosForTokenClassification,
            BrosModel,
            BrosPreTrainedModel,
            BrosProcessor,
            BrosSpadeEEForTokenClassification,
            BrosSpadeELForTokenClassification,
        )
        from transformers.src.transformers.models.camembert import (
            CamembertForCausalLM,
            CamembertForMaskedLM,
            CamembertForMultipleChoice,
            CamembertForQuestionAnswering,
            CamembertForSequenceClassification,
            CamembertForTokenClassification,
            CamembertModel,
            CamembertPreTrainedModel,
        )
        from transformers.src.transformers.models.canine import (
            CanineForMultipleChoice,
            CanineForQuestionAnswering,
            CanineForSequenceClassification,
            CanineForTokenClassification,
            CanineModel,
            CaninePreTrainedModel,
            load_tf_weights_in_canine,
        )
        from transformers.src.transformers.models.chameleon import (
            ChameleonForConditionalGeneration,
            ChameleonModel,
            ChameleonPreTrainedModel,
            ChameleonProcessor,
            ChameleonVQVAE,
        )
        from transformers.src.transformers.models.chinese_clip import (
            ChineseCLIPModel,
            ChineseCLIPPreTrainedModel,
            ChineseCLIPTextModel,
            ChineseCLIPVisionModel,
        )
        from transformers.src.transformers.models.clap import (
            ClapAudioModel,
            ClapAudioModelWithProjection,
            ClapFeatureExtractor,
            ClapModel,
            ClapPreTrainedModel,
            ClapTextModel,
            ClapTextModelWithProjection,
        )
        from transformers.src.transformers.models.clip import (
            CLIPForImageClassification,
            CLIPModel,
            CLIPPreTrainedModel,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModel,
            CLIPVisionModelWithProjection,
        )
        from transformers.src.transformers.models.clipseg import (
            CLIPSegForImageSegmentation,
            CLIPSegModel,
            CLIPSegPreTrainedModel,
            CLIPSegTextModel,
            CLIPSegVisionModel,
        )
        from transformers.src.transformers.models.clvp import (
            ClvpDecoder,
            ClvpEncoder,
            ClvpForCausalLM,
            ClvpModel,
            ClvpModelForConditionalGeneration,
            ClvpPreTrainedModel,
        )
        from transformers.src.transformers.models.codegen import (
            CodeGenForCausalLM,
            CodeGenModel,
            CodeGenPreTrainedModel,
        )
        from transformers.src.transformers.models.cohere import (
            CohereForCausalLM,
            CohereModel,
            CoherePreTrainedModel,
        )
        from transformers.src.transformers.models.conditional_detr import (
            ConditionalDetrForObjectDetection,
            ConditionalDetrForSegmentation,
            ConditionalDetrModel,
            ConditionalDetrPreTrainedModel,
        )
        from transformers.src.transformers.models.convbert import (
            ConvBertForMaskedLM,
            ConvBertForMultipleChoice,
            ConvBertForQuestionAnswering,
            ConvBertForSequenceClassification,
            ConvBertForTokenClassification,
            ConvBertModel,
            ConvBertPreTrainedModel,
            load_tf_weights_in_convbert,
        )
        from transformers.src.transformers.models.convnext import (
            ConvNextBackbone,
            ConvNextForImageClassification,
            ConvNextModel,
            ConvNextPreTrainedModel,
        )
        from transformers.src.transformers.models.convnextv2 import (
            ConvNextV2Backbone,
            ConvNextV2ForImageClassification,
            ConvNextV2Model,
            ConvNextV2PreTrainedModel,
        )
        from transformers.src.transformers.models.cpmant import (
            CpmAntForCausalLM,
            CpmAntModel,
            CpmAntPreTrainedModel,
        )
        from transformers.src.transformers.models.ctrl import (
            CTRLForSequenceClassification,
            CTRLLMHeadModel,
            CTRLModel,
            CTRLPreTrainedModel,
        )
        from transformers.src.transformers.models.cvt import (
            CvtForImageClassification,
            CvtModel,
            CvtPreTrainedModel,
        )
        from transformers.src.transformers.models.dac import (
            DacModel,
            DacPreTrainedModel,
        )
        from transformers.src.transformers.models.data2vec import (
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextModel,
            Data2VecTextPreTrainedModel,
            Data2VecVisionForImageClassification,
            Data2VecVisionForSemanticSegmentation,
            Data2VecVisionModel,
            Data2VecVisionPreTrainedModel,
        )

        # PyTorch model imports
        from transformers.src.transformers.models.dbrx import (
            DbrxForCausalLM,
            DbrxModel,
            DbrxPreTrainedModel,
        )
        from transformers.src.transformers.models.deberta import (
            DebertaForMaskedLM,
            DebertaForQuestionAnswering,
            DebertaForSequenceClassification,
            DebertaForTokenClassification,
            DebertaModel,
            DebertaPreTrainedModel,
        )
        from transformers.src.transformers.models.deberta_v2 import (
            DebertaV2ForMaskedLM,
            DebertaV2ForMultipleChoice,
            DebertaV2ForQuestionAnswering,
            DebertaV2ForSequenceClassification,
            DebertaV2ForTokenClassification,
            DebertaV2Model,
            DebertaV2PreTrainedModel,
        )
        from transformers.src.transformers.models.decision_transformer import (
            DecisionTransformerGPT2Model,
            DecisionTransformerGPT2PreTrainedModel,
            DecisionTransformerModel,
            DecisionTransformerPreTrainedModel,
        )
        from transformers.src.transformers.models.deformable_detr import (
            DeformableDetrForObjectDetection,
            DeformableDetrModel,
            DeformableDetrPreTrainedModel,
        )
        from transformers.src.transformers.models.deit import (
            DeiTForImageClassification,
            DeiTForImageClassificationWithTeacher,
            DeiTForMaskedImageModeling,
            DeiTModel,
            DeiTPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.deta import (
            DetaForObjectDetection,
            DetaModel,
            DetaPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.efficientformer import (
            EfficientFormerForImageClassification,
            EfficientFormerForImageClassificationWithTeacher,
            EfficientFormerModel,
            EfficientFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.ernie_m import (
            ErnieMForInformationExtraction,
            ErnieMForMultipleChoice,
            ErnieMForQuestionAnswering,
            ErnieMForSequenceClassification,
            ErnieMForTokenClassification,
            ErnieMModel,
            ErnieMPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.gptsan_japanese import (
            GPTSanJapaneseForConditionalGeneration,
            GPTSanJapaneseModel,
            GPTSanJapanesePreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.graphormer import (
            GraphormerForGraphClassification,
            GraphormerModel,
            GraphormerPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.jukebox import (
            JukeboxModel,
            JukeboxPreTrainedModel,
            JukeboxPrior,
            JukeboxVQVAE,
        )
        from transformers.src.transformers.models.deprecated.mctct import (
            MCTCTForCTC,
            MCTCTModel,
            MCTCTPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.mega import (
            MegaForCausalLM,
            MegaForMaskedLM,
            MegaForMultipleChoice,
            MegaForQuestionAnswering,
            MegaForSequenceClassification,
            MegaForTokenClassification,
            MegaModel,
            MegaPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.mmbt import (
            MMBTForClassification,
            MMBTModel,
            ModalEmbeddings,
        )
        from transformers.src.transformers.models.deprecated.nat import (
            NatBackbone,
            NatForImageClassification,
            NatModel,
            NatPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.nezha import (
            NezhaForMaskedLM,
            NezhaForMultipleChoice,
            NezhaForNextSentencePrediction,
            NezhaForPreTraining,
            NezhaForQuestionAnswering,
            NezhaForSequenceClassification,
            NezhaForTokenClassification,
            NezhaModel,
            NezhaPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.open_llama import (
            OpenLlamaForCausalLM,
            OpenLlamaForSequenceClassification,
            OpenLlamaModel,
            OpenLlamaPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.qdqbert import (
            QDQBertForMaskedLM,
            QDQBertForMultipleChoice,
            QDQBertForNextSentencePrediction,
            QDQBertForQuestionAnswering,
            QDQBertForSequenceClassification,
            QDQBertForTokenClassification,
            QDQBertLMHeadModel,
            QDQBertModel,
            QDQBertPreTrainedModel,
            load_tf_weights_in_qdqbert,
        )
        from transformers.src.transformers.models.deprecated.realm import (
            RealmEmbedder,
            RealmForOpenQA,
            RealmKnowledgeAugEncoder,
            RealmPreTrainedModel,
            RealmReader,
            RealmRetriever,
            RealmScorer,
            load_tf_weights_in_realm,
        )
        from transformers.src.transformers.models.deprecated.retribert import (
            RetriBertModel,
            RetriBertPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.speech_to_text_2 import (
            Speech2Text2ForCausalLM,
            Speech2Text2PreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.trajectory_transformer import (
            TrajectoryTransformerModel,
            TrajectoryTransformerPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.transfo_xl import (
            AdaptiveEmbedding,
            TransfoXLForSequenceClassification,
            TransfoXLLMHeadModel,
            TransfoXLModel,
            TransfoXLPreTrainedModel,
            load_tf_weights_in_transfo_xl,
        )
        from transformers.src.transformers.models.deprecated.tvlt import (
            TvltForAudioVisualClassification,
            TvltForPreTraining,
            TvltModel,
            TvltPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.van import (
            VanForImageClassification,
            VanModel,
            VanPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.vit_hybrid import (
            ViTHybridForImageClassification,
            ViTHybridModel,
            ViTHybridPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.xlm_prophetnet import (
            XLMProphetNetDecoder,
            XLMProphetNetEncoder,
            XLMProphetNetForCausalLM,
            XLMProphetNetForConditionalGeneration,
            XLMProphetNetModel,
            XLMProphetNetPreTrainedModel,
        )
        from transformers.src.transformers.models.depth_anything import (
            DepthAnythingForDepthEstimation,
            DepthAnythingPreTrainedModel,
        )
        from transformers.src.transformers.models.detr import (
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )
        from transformers.src.transformers.models.dinat import (
            DinatBackbone,
            DinatForImageClassification,
            DinatModel,
            DinatPreTrainedModel,
        )
        from transformers.src.transformers.models.dinov2 import (
            Dinov2Backbone,
            Dinov2ForImageClassification,
            Dinov2Model,
            Dinov2PreTrainedModel,
        )
        from transformers.src.transformers.models.distilbert import (
            DistilBertForMaskedLM,
            DistilBertForMultipleChoice,
            DistilBertForQuestionAnswering,
            DistilBertForSequenceClassification,
            DistilBertForTokenClassification,
            DistilBertModel,
            DistilBertPreTrainedModel,
        )
        from transformers.src.transformers.models.donut import (
            DonutSwinModel,
            DonutSwinPreTrainedModel,
        )
        from transformers.src.transformers.models.dpr import (
            DPRContextEncoder,
            DPRPretrainedContextEncoder,
            DPRPreTrainedModel,
            DPRPretrainedQuestionEncoder,
            DPRPretrainedReader,
            DPRQuestionEncoder,
            DPRReader,
        )
        from transformers.src.transformers.models.dpt import (
            DPTForDepthEstimation,
            DPTForSemanticSegmentation,
            DPTModel,
            DPTPreTrainedModel,
        )
        from transformers.src.transformers.models.efficientnet import (
            EfficientNetForImageClassification,
            EfficientNetModel,
            EfficientNetPreTrainedModel,
        )
        from transformers.src.transformers.models.electra import (
            ElectraForCausalLM,
            ElectraForMaskedLM,
            ElectraForMultipleChoice,
            ElectraForPreTraining,
            ElectraForQuestionAnswering,
            ElectraForSequenceClassification,
            ElectraForTokenClassification,
            ElectraModel,
            ElectraPreTrainedModel,
            load_tf_weights_in_electra,
        )
        from transformers.src.transformers.models.encodec import (
            EncodecModel,
            EncodecPreTrainedModel,
        )
        from transformers.src.transformers.models.encoder_decoder import EncoderDecoderModel
        from transformers.src.transformers.models.ernie import (
            ErnieForCausalLM,
            ErnieForMaskedLM,
            ErnieForMultipleChoice,
            ErnieForNextSentencePrediction,
            ErnieForPreTraining,
            ErnieForQuestionAnswering,
            ErnieForSequenceClassification,
            ErnieForTokenClassification,
            ErnieModel,
            ErniePreTrainedModel,
        )
        from transformers.src.transformers.models.esm import (
            EsmFoldPreTrainedModel,
            EsmForMaskedLM,
            EsmForProteinFolding,
            EsmForSequenceClassification,
            EsmForTokenClassification,
            EsmModel,
            EsmPreTrainedModel,
        )
        from transformers.src.transformers.models.falcon import (
            FalconForCausalLM,
            FalconForQuestionAnswering,
            FalconForSequenceClassification,
            FalconForTokenClassification,
            FalconModel,
            FalconPreTrainedModel,
        )
        from transformers.src.transformers.models.falcon_mamba import (
            FalconMambaForCausalLM,
            FalconMambaModel,
            FalconMambaPreTrainedModel,
        )
        from transformers.src.transformers.models.fastspeech2_conformer import (
            FastSpeech2ConformerHifiGan,
            FastSpeech2ConformerModel,
            FastSpeech2ConformerPreTrainedModel,
            FastSpeech2ConformerWithHifiGan,
        )
        from transformers.src.transformers.models.flaubert import (
            FlaubertForMultipleChoice,
            FlaubertForQuestionAnswering,
            FlaubertForQuestionAnsweringSimple,
            FlaubertForSequenceClassification,
            FlaubertForTokenClassification,
            FlaubertModel,
            FlaubertPreTrainedModel,
            FlaubertWithLMHeadModel,
        )
        from transformers.src.transformers.models.flava import (
            FlavaForPreTraining,
            FlavaImageCodebook,
            FlavaImageModel,
            FlavaModel,
            FlavaMultimodalModel,
            FlavaPreTrainedModel,
            FlavaTextModel,
        )
        from transformers.src.transformers.models.fnet import (
            FNetForMaskedLM,
            FNetForMultipleChoice,
            FNetForNextSentencePrediction,
            FNetForPreTraining,
            FNetForQuestionAnswering,
            FNetForSequenceClassification,
            FNetForTokenClassification,
            FNetModel,
            FNetPreTrainedModel,
        )
        from transformers.src.transformers.models.focalnet import (
            FocalNetBackbone,
            FocalNetForImageClassification,
            FocalNetForMaskedImageModeling,
            FocalNetModel,
            FocalNetPreTrainedModel,
        )
        from transformers.src.transformers.models.fsmt import (
            FSMTForConditionalGeneration,
            FSMTModel,
            PretrainedFSMTModel,
        )
        from transformers.src.transformers.models.funnel import (
            FunnelBaseModel,
            FunnelForMaskedLM,
            FunnelForMultipleChoice,
            FunnelForPreTraining,
            FunnelForQuestionAnswering,
            FunnelForSequenceClassification,
            FunnelForTokenClassification,
            FunnelModel,
            FunnelPreTrainedModel,
            load_tf_weights_in_funnel,
        )
        from transformers.src.transformers.models.fuyu import (
            FuyuForCausalLM,
            FuyuPreTrainedModel,
        )
        from transformers.src.transformers.models.gemma import (
            GemmaForCausalLM,
            GemmaForSequenceClassification,
            GemmaForTokenClassification,
            GemmaModel,
            GemmaPreTrainedModel,
        )
        from transformers.src.transformers.models.gemma2 import (
            Gemma2ForCausalLM,
            Gemma2ForSequenceClassification,
            Gemma2ForTokenClassification,
            Gemma2Model,
            Gemma2PreTrainedModel,
        )
        from transformers.src.transformers.models.git import (
            GitForCausalLM,
            GitModel,
            GitPreTrainedModel,
            GitVisionModel,
        )
        from transformers.src.transformers.models.glm import (
            GlmForCausalLM,
            GlmForSequenceClassification,
            GlmForTokenClassification,
            GlmModel,
            GlmPreTrainedModel,
        )
        from transformers.src.transformers.models.glpn import (
            GLPNForDepthEstimation,
            GLPNModel,
            GLPNPreTrainedModel,
        )
        from transformers.src.transformers.models.gpt2 import (
            GPT2DoubleHeadsModel,
            GPT2ForQuestionAnswering,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
            GPT2LMHeadModel,
            GPT2Model,
            GPT2PreTrainedModel,
            load_tf_weights_in_gpt2,
        )
        from transformers.src.transformers.models.gpt_bigcode import (
            GPTBigCodeForCausalLM,
            GPTBigCodeForSequenceClassification,
            GPTBigCodeForTokenClassification,
            GPTBigCodeModel,
            GPTBigCodePreTrainedModel,
        )
        from transformers.src.transformers.models.gpt_neo import (
            GPTNeoForCausalLM,
            GPTNeoForQuestionAnswering,
            GPTNeoForSequenceClassification,
            GPTNeoForTokenClassification,
            GPTNeoModel,
            GPTNeoPreTrainedModel,
            load_tf_weights_in_gpt_neo,
        )
        from transformers.src.transformers.models.gpt_neox import (
            GPTNeoXForCausalLM,
            GPTNeoXForQuestionAnswering,
            GPTNeoXForSequenceClassification,
            GPTNeoXForTokenClassification,
            GPTNeoXModel,
            GPTNeoXPreTrainedModel,
        )
        from transformers.src.transformers.models.gpt_neox_japanese import (
            GPTNeoXJapaneseForCausalLM,
            GPTNeoXJapaneseModel,
            GPTNeoXJapanesePreTrainedModel,
        )
        from transformers.src.transformers.models.gptj import (
            GPTJForCausalLM,
            GPTJForQuestionAnswering,
            GPTJForSequenceClassification,
            GPTJModel,
            GPTJPreTrainedModel,
        )
        from transformers.src.transformers.models.granite import (
            GraniteForCausalLM,
            GraniteModel,
            GranitePreTrainedModel,
        )
        from transformers.src.transformers.models.granitemoe import (
            GraniteMoeForCausalLM,
            GraniteMoeModel,
            GraniteMoePreTrainedModel,
        )
        from transformers.src.transformers.models.grounding_dino import (
            GroundingDinoForObjectDetection,
            GroundingDinoModel,
            GroundingDinoPreTrainedModel,
        )
        from transformers.src.transformers.models.groupvit import (
            GroupViTModel,
            GroupViTPreTrainedModel,
            GroupViTTextModel,
            GroupViTVisionModel,
        )
        from transformers.src.transformers.models.hiera import (
            HieraBackbone,
            HieraForImageClassification,
            HieraForPreTraining,
            HieraModel,
            HieraPreTrainedModel,
        )
        from transformers.src.transformers.models.hubert import (
            HubertForCTC,
            HubertForSequenceClassification,
            HubertModel,
            HubertPreTrainedModel,
        )
        from transformers.src.transformers.models.ibert import (
            IBertForMaskedLM,
            IBertForMultipleChoice,
            IBertForQuestionAnswering,
            IBertForSequenceClassification,
            IBertForTokenClassification,
            IBertModel,
            IBertPreTrainedModel,
        )
        from transformers.src.transformers.models.idefics import (
            IdeficsForVisionText2Text,
            IdeficsModel,
            IdeficsPreTrainedModel,
            IdeficsProcessor,
        )
        from transformers.src.transformers.models.idefics2 import (
            Idefics2ForConditionalGeneration,
            Idefics2Model,
            Idefics2PreTrainedModel,
            Idefics2Processor,
        )
        from transformers.src.transformers.models.idefics3 import (
            Idefics3ForConditionalGeneration,
            Idefics3Model,
            Idefics3PreTrainedModel,
            Idefics3Processor,
        )
        from transformers.src.transformers.models.imagegpt import (
            ImageGPTForCausalImageModeling,
            ImageGPTForImageClassification,
            ImageGPTModel,
            ImageGPTPreTrainedModel,
            load_tf_weights_in_imagegpt,
        )
        from transformers.src.transformers.models.informer import (
            InformerForPrediction,
            InformerModel,
            InformerPreTrainedModel,
        )
        from transformers.src.transformers.models.instructblip import (
            InstructBlipForConditionalGeneration,
            InstructBlipPreTrainedModel,
            InstructBlipQFormerModel,
            InstructBlipVisionModel,
        )
        from transformers.src.transformers.models.instructblipvideo import (
            InstructBlipVideoForConditionalGeneration,
            InstructBlipVideoPreTrainedModel,
            InstructBlipVideoQFormerModel,
            InstructBlipVideoVisionModel,
        )
        from transformers.src.transformers.models.jamba import (
            JambaForCausalLM,
            JambaForSequenceClassification,
            JambaModel,
            JambaPreTrainedModel,
        )
        from transformers.src.transformers.models.jetmoe import (
            JetMoeForCausalLM,
            JetMoeForSequenceClassification,
            JetMoeModel,
            JetMoePreTrainedModel,
        )
        from transformers.src.transformers.models.kosmos2 import (
            Kosmos2ForConditionalGeneration,
            Kosmos2Model,
            Kosmos2PreTrainedModel,
        )
        from transformers.src.transformers.models.layoutlm import (
            LayoutLMForMaskedLM,
            LayoutLMForQuestionAnswering,
            LayoutLMForSequenceClassification,
            LayoutLMForTokenClassification,
            LayoutLMModel,
            LayoutLMPreTrainedModel,
        )
        from transformers.src.transformers.models.layoutlmv2 import (
            LayoutLMv2ForQuestionAnswering,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2Model,
            LayoutLMv2PreTrainedModel,
        )
        from transformers.src.transformers.models.layoutlmv3 import (
            LayoutLMv3ForQuestionAnswering,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Model,
            LayoutLMv3PreTrainedModel,
        )
        from transformers.src.transformers.models.led import (
            LEDForConditionalGeneration,
            LEDForQuestionAnswering,
            LEDForSequenceClassification,
            LEDModel,
            LEDPreTrainedModel,
        )
        from transformers.src.transformers.models.levit import (
            LevitForImageClassification,
            LevitForImageClassificationWithTeacher,
            LevitModel,
            LevitPreTrainedModel,
        )
        from transformers.src.transformers.models.lilt import (
            LiltForQuestionAnswering,
            LiltForSequenceClassification,
            LiltForTokenClassification,
            LiltModel,
            LiltPreTrainedModel,
        )
        from transformers.src.transformers.models.llama import (
            LlamaForCausalLM,
            LlamaForQuestionAnswering,
            LlamaForSequenceClassification,
            LlamaForTokenClassification,
            LlamaModel,
            LlamaPreTrainedModel,
        )
        from transformers.src.transformers.models.llava import (
            LlavaForConditionalGeneration,
            LlavaPreTrainedModel,
        )
        from transformers.src.transformers.models.llava_next import (
            LlavaNextForConditionalGeneration,
            LlavaNextPreTrainedModel,
        )
        from transformers.src.transformers.models.llava_next_video import (
            LlavaNextVideoForConditionalGeneration,
            LlavaNextVideoPreTrainedModel,
        )
        from transformers.src.transformers.models.llava_onevision import (
            LlavaOnevisionForConditionalGeneration,
            LlavaOnevisionPreTrainedModel,
        )
        from transformers.src.transformers.models.longformer import (
            LongformerForMaskedLM,
            LongformerForMultipleChoice,
            LongformerForQuestionAnswering,
            LongformerForSequenceClassification,
            LongformerForTokenClassification,
            LongformerModel,
            LongformerPreTrainedModel,
        )
        from transformers.src.transformers.models.longt5 import (
            LongT5EncoderModel,
            LongT5ForConditionalGeneration,
            LongT5Model,
            LongT5PreTrainedModel,
        )
        from transformers.src.transformers.models.luke import (
            LukeForEntityClassification,
            LukeForEntityPairClassification,
            LukeForEntitySpanClassification,
            LukeForMaskedLM,
            LukeForMultipleChoice,
            LukeForQuestionAnswering,
            LukeForSequenceClassification,
            LukeForTokenClassification,
            LukeModel,
            LukePreTrainedModel,
        )
        from transformers.src.transformers.models.lxmert import (
            LxmertEncoder,
            LxmertForPreTraining,
            LxmertForQuestionAnswering,
            LxmertModel,
            LxmertPreTrainedModel,
            LxmertVisualFeatureEncoder,
        )
        from transformers.src.transformers.models.m2m_100 import (
            M2M100ForConditionalGeneration,
            M2M100Model,
            M2M100PreTrainedModel,
        )
        from transformers.src.transformers.models.mamba import (
            MambaForCausalLM,
            MambaModel,
            MambaPreTrainedModel,
        )
        from transformers.src.transformers.models.mamba2 import (
            Mamba2ForCausalLM,
            Mamba2Model,
            Mamba2PreTrainedModel,
        )
        from transformers.src.transformers.models.marian import MarianForCausalLM, MarianModel, MarianMTModel, MarianPreTrainedModel
        from transformers.src.transformers.models.markuplm import (
            MarkupLMForQuestionAnswering,
            MarkupLMForSequenceClassification,
            MarkupLMForTokenClassification,
            MarkupLMModel,
            MarkupLMPreTrainedModel,
        )
        from transformers.src.transformers.models.mask2former import (
            Mask2FormerForUniversalSegmentation,
            Mask2FormerModel,
            Mask2FormerPreTrainedModel,
        )
        from transformers.src.transformers.models.maskformer import (
            MaskFormerForInstanceSegmentation,
            MaskFormerModel,
            MaskFormerPreTrainedModel,
            MaskFormerSwinBackbone,
        )
        from transformers.src.transformers.models.mbart import (
            MBartForCausalLM,
            MBartForConditionalGeneration,
            MBartForQuestionAnswering,
            MBartForSequenceClassification,
            MBartModel,
            MBartPreTrainedModel,
        )
        from transformers.src.transformers.models.megatron_bert import (
            MegatronBertForCausalLM,
            MegatronBertForMaskedLM,
            MegatronBertForMultipleChoice,
            MegatronBertForNextSentencePrediction,
            MegatronBertForPreTraining,
            MegatronBertForQuestionAnswering,
            MegatronBertForSequenceClassification,
            MegatronBertForTokenClassification,
            MegatronBertModel,
            MegatronBertPreTrainedModel,
        )
        from transformers.src.transformers.models.mgp_str import (
            MgpstrForSceneTextRecognition,
            MgpstrModel,
            MgpstrPreTrainedModel,
        )
        from transformers.src.transformers.models.mimi import (
            MimiModel,
            MimiPreTrainedModel,
        )
        from transformers.src.transformers.models.mistral import (
            MistralForCausalLM,
            MistralForQuestionAnswering,
            MistralForSequenceClassification,
            MistralForTokenClassification,
            MistralModel,
            MistralPreTrainedModel,
        )
        from transformers.src.transformers.models.mixtral import (
            MixtralForCausalLM,
            MixtralForQuestionAnswering,
            MixtralForSequenceClassification,
            MixtralForTokenClassification,
            MixtralModel,
            MixtralPreTrainedModel,
        )
        from transformers.src.transformers.models.mllama import (
            MllamaForCausalLM,
            MllamaForConditionalGeneration,
            MllamaPreTrainedModel,
            MllamaProcessor,
            MllamaTextModel,
            MllamaVisionModel,
        )
        from transformers.src.transformers.models.mobilebert import (
            MobileBertForMaskedLM,
            MobileBertForMultipleChoice,
            MobileBertForNextSentencePrediction,
            MobileBertForPreTraining,
            MobileBertForQuestionAnswering,
            MobileBertForSequenceClassification,
            MobileBertForTokenClassification,
            MobileBertModel,
            MobileBertPreTrainedModel,
            load_tf_weights_in_mobilebert,
        )
        from transformers.src.transformers.models.mobilenet_v1 import (
            MobileNetV1ForImageClassification,
            MobileNetV1Model,
            MobileNetV1PreTrainedModel,
            load_tf_weights_in_mobilenet_v1,
        )
        from transformers.src.transformers.models.mobilenet_v2 import (
            MobileNetV2ForImageClassification,
            MobileNetV2ForSemanticSegmentation,
            MobileNetV2Model,
            MobileNetV2PreTrainedModel,
            load_tf_weights_in_mobilenet_v2,
        )
        from transformers.src.transformers.models.mobilevit import (
            MobileViTForImageClassification,
            MobileViTForSemanticSegmentation,
            MobileViTModel,
            MobileViTPreTrainedModel,
        )
        from transformers.src.transformers.models.mobilevitv2 import (
            MobileViTV2ForImageClassification,
            MobileViTV2ForSemanticSegmentation,
            MobileViTV2Model,
            MobileViTV2PreTrainedModel,
        )
        from transformers.src.transformers.models.moshi import (
            MoshiForCausalLM,
            MoshiForConditionalGeneration,
            MoshiModel,
            MoshiPreTrainedModel,
        )
        from transformers.src.transformers.models.mpnet import (
            MPNetForMaskedLM,
            MPNetForMultipleChoice,
            MPNetForQuestionAnswering,
            MPNetForSequenceClassification,
            MPNetForTokenClassification,
            MPNetModel,
            MPNetPreTrainedModel,
        )
        from transformers.src.transformers.models.mpt import (
            MptForCausalLM,
            MptForQuestionAnswering,
            MptForSequenceClassification,
            MptForTokenClassification,
            MptModel,
            MptPreTrainedModel,
        )
        from transformers.src.transformers.models.mra import (
            MraForMaskedLM,
            MraForMultipleChoice,
            MraForQuestionAnswering,
            MraForSequenceClassification,
            MraForTokenClassification,
            MraModel,
            MraPreTrainedModel,
        )
        from transformers.src.transformers.models.mt5 import (
            MT5EncoderModel,
            MT5ForConditionalGeneration,
            MT5ForQuestionAnswering,
            MT5ForSequenceClassification,
            MT5ForTokenClassification,
            MT5Model,
            MT5PreTrainedModel,
        )
        from transformers.src.transformers.models.musicgen import (
            MusicgenForCausalLM,
            MusicgenForConditionalGeneration,
            MusicgenModel,
            MusicgenPreTrainedModel,
            MusicgenProcessor,
        )
        from transformers.src.transformers.models.musicgen_melody import (
            MusicgenMelodyForCausalLM,
            MusicgenMelodyForConditionalGeneration,
            MusicgenMelodyModel,
            MusicgenMelodyPreTrainedModel,
        )
        from transformers.src.transformers.models.mvp import (
            MvpForCausalLM,
            MvpForConditionalGeneration,
            MvpForQuestionAnswering,
            MvpForSequenceClassification,
            MvpModel,
            MvpPreTrainedModel,
        )
        from transformers.src.transformers.models.nemotron import (
            NemotronForCausalLM,
            NemotronForQuestionAnswering,
            NemotronForSequenceClassification,
            NemotronForTokenClassification,
            NemotronModel,
            NemotronPreTrainedModel,
        )
        from transformers.src.transformers.models.nllb_moe import (
            NllbMoeForConditionalGeneration,
            NllbMoeModel,
            NllbMoePreTrainedModel,
            NllbMoeSparseMLP,
            NllbMoeTop2Router,
        )
        from transformers.src.transformers.models.nystromformer import (
            NystromformerForMaskedLM,
            NystromformerForMultipleChoice,
            NystromformerForQuestionAnswering,
            NystromformerForSequenceClassification,
            NystromformerForTokenClassification,
            NystromformerModel,
            NystromformerPreTrainedModel,
        )
        from transformers.src.transformers.models.olmo import (
            OlmoForCausalLM,
            OlmoModel,
            OlmoPreTrainedModel,
        )
        from transformers.src.transformers.models.olmoe import (
            OlmoeForCausalLM,
            OlmoeModel,
            OlmoePreTrainedModel,
        )
        from transformers.src.transformers.models.omdet_turbo import (
            OmDetTurboForObjectDetection,
            OmDetTurboPreTrainedModel,
        )
        from transformers.src.transformers.models.oneformer import (
            OneFormerForUniversalSegmentation,
            OneFormerModel,
            OneFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.openai import (
            OpenAIGPTDoubleHeadsModel,
            OpenAIGPTForSequenceClassification,
            OpenAIGPTLMHeadModel,
            OpenAIGPTModel,
            OpenAIGPTPreTrainedModel,
            load_tf_weights_in_openai_gpt,
        )
        from transformers.src.transformers.models.opt import (
            OPTForCausalLM,
            OPTForQuestionAnswering,
            OPTForSequenceClassification,
            OPTModel,
            OPTPreTrainedModel,
        )
        from transformers.src.transformers.models.owlv2 import (
            Owlv2ForObjectDetection,
            Owlv2Model,
            Owlv2PreTrainedModel,
            Owlv2TextModel,
            Owlv2VisionModel,
        )
        from transformers.src.transformers.models.owlvit import (
            OwlViTForObjectDetection,
            OwlViTModel,
            OwlViTPreTrainedModel,
            OwlViTTextModel,
            OwlViTVisionModel,
        )
        from transformers.src.transformers.models.paligemma import (
            PaliGemmaForConditionalGeneration,
            PaliGemmaPreTrainedModel,
            PaliGemmaProcessor,
        )
        from transformers.src.transformers.models.patchtsmixer import (
            PatchTSMixerForPrediction,
            PatchTSMixerForPretraining,
            PatchTSMixerForRegression,
            PatchTSMixerForTimeSeriesClassification,
            PatchTSMixerModel,
            PatchTSMixerPreTrainedModel,
        )
        from transformers.src.transformers.models.patchtst import (
            PatchTSTForClassification,
            PatchTSTForPrediction,
            PatchTSTForPretraining,
            PatchTSTForRegression,
            PatchTSTModel,
            PatchTSTPreTrainedModel,
        )
        from transformers.src.transformers.models.pegasus import (
            PegasusForCausalLM,
            PegasusForConditionalGeneration,
            PegasusModel,
            PegasusPreTrainedModel,
        )
        from transformers.src.transformers.models.pegasus_x import (
            PegasusXForConditionalGeneration,
            PegasusXModel,
            PegasusXPreTrainedModel,
        )
        from transformers.src.transformers.models.perceiver import (
            PerceiverForImageClassificationConvProcessing,
            PerceiverForImageClassificationFourier,
            PerceiverForImageClassificationLearned,
            PerceiverForMaskedLM,
            PerceiverForMultimodalAutoencoding,
            PerceiverForOpticalFlow,
            PerceiverForSequenceClassification,
            PerceiverModel,
            PerceiverPreTrainedModel,
        )
        from transformers.src.transformers.models.persimmon import (
            PersimmonForCausalLM,
            PersimmonForSequenceClassification,
            PersimmonForTokenClassification,
            PersimmonModel,
            PersimmonPreTrainedModel,
        )
        from transformers.src.transformers.models.phi import (
            PhiForCausalLM,
            PhiForSequenceClassification,
            PhiForTokenClassification,
            PhiModel,
            PhiPreTrainedModel,
        )
        from transformers.src.transformers.models.phi3 import (
            Phi3ForCausalLM,
            Phi3ForSequenceClassification,
            Phi3ForTokenClassification,
            Phi3Model,
            Phi3PreTrainedModel,
        )
        from transformers.src.transformers.models.phimoe import (
            PhimoeForCausalLM,
            PhimoeForSequenceClassification,
            PhimoeModel,
            PhimoePreTrainedModel,
        )
        from transformers.src.transformers.models.pix2struct import (
            Pix2StructForConditionalGeneration,
            Pix2StructPreTrainedModel,
            Pix2StructTextModel,
            Pix2StructVisionModel,
        )
        from transformers.src.transformers.models.pixtral import (
            PixtralPreTrainedModel,
            PixtralVisionModel,
        )
        from transformers.src.transformers.models.plbart import (
            PLBartForCausalLM,
            PLBartForConditionalGeneration,
            PLBartForSequenceClassification,
            PLBartModel,
            PLBartPreTrainedModel,
        )
        from transformers.src.transformers.models.poolformer import (
            PoolFormerForImageClassification,
            PoolFormerModel,
            PoolFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.pop2piano import (
            Pop2PianoForConditionalGeneration,
            Pop2PianoPreTrainedModel,
        )
        from transformers.src.transformers.models.prophetnet import (
            ProphetNetDecoder,
            ProphetNetEncoder,
            ProphetNetForCausalLM,
            ProphetNetForConditionalGeneration,
            ProphetNetModel,
            ProphetNetPreTrainedModel,
        )
        from transformers.src.transformers.models.pvt import (
            PvtForImageClassification,
            PvtModel,
            PvtPreTrainedModel,
        )
        from transformers.src.transformers.models.pvt_v2 import (
            PvtV2Backbone,
            PvtV2ForImageClassification,
            PvtV2Model,
            PvtV2PreTrainedModel,
        )
        from transformers.src.transformers.models.qwen2 import (
            Qwen2ForCausalLM,
            Qwen2ForQuestionAnswering,
            Qwen2ForSequenceClassification,
            Qwen2ForTokenClassification,
            Qwen2Model,
            Qwen2PreTrainedModel,
        )
        from transformers.src.transformers.models.qwen2_audio import (
            Qwen2AudioEncoder,
            Qwen2AudioForConditionalGeneration,
            Qwen2AudioPreTrainedModel,
        )
        from transformers.src.transformers.models.qwen2_moe import (
            Qwen2MoeForCausalLM,
            Qwen2MoeForQuestionAnswering,
            Qwen2MoeForSequenceClassification,
            Qwen2MoeForTokenClassification,
            Qwen2MoeModel,
            Qwen2MoePreTrainedModel,
        )
        from transformers.src.transformers.models.qwen2_vl import (
            Qwen2VLForConditionalGeneration,
            Qwen2VLModel,
            Qwen2VLPreTrainedModel,
        )
        from transformers.src.transformers.models.rag import (
            RagModel,
            RagPreTrainedModel,
            RagSequenceForGeneration,
            RagTokenForGeneration,
        )
        from transformers.src.transformers.models.recurrent_gemma import (
            RecurrentGemmaForCausalLM,
            RecurrentGemmaModel,
            RecurrentGemmaPreTrainedModel,
        )
        from transformers.src.transformers.models.reformer import (
            ReformerForMaskedLM,
            ReformerForQuestionAnswering,
            ReformerForSequenceClassification,
            ReformerModel,
            ReformerModelWithLMHead,
            ReformerPreTrainedModel,
        )
        from transformers.src.transformers.models.regnet import (
            RegNetForImageClassification,
            RegNetModel,
            RegNetPreTrainedModel,
        )
        from transformers.src.transformers.models.rembert import (
            RemBertForCausalLM,
            RemBertForMaskedLM,
            RemBertForMultipleChoice,
            RemBertForQuestionAnswering,
            RemBertForSequenceClassification,
            RemBertForTokenClassification,
            RemBertModel,
            RemBertPreTrainedModel,
            load_tf_weights_in_rembert,
        )
        from transformers.src.transformers.models.resnet import (
            ResNetBackbone,
            ResNetForImageClassification,
            ResNetModel,
            ResNetPreTrainedModel,
        )
        from transformers.src.transformers.models.roberta import (
            RobertaForCausalLM,
            RobertaForMaskedLM,
            RobertaForMultipleChoice,
            RobertaForQuestionAnswering,
            RobertaForSequenceClassification,
            RobertaForTokenClassification,
            RobertaModel,
            RobertaPreTrainedModel,
        )
        from transformers.src.transformers.models.roberta_prelayernorm import (
            RobertaPreLayerNormForCausalLM,
            RobertaPreLayerNormForMaskedLM,
            RobertaPreLayerNormForMultipleChoice,
            RobertaPreLayerNormForQuestionAnswering,
            RobertaPreLayerNormForSequenceClassification,
            RobertaPreLayerNormForTokenClassification,
            RobertaPreLayerNormModel,
            RobertaPreLayerNormPreTrainedModel,
        )
        from transformers.src.transformers.models.roc_bert import (
            RoCBertForCausalLM,
            RoCBertForMaskedLM,
            RoCBertForMultipleChoice,
            RoCBertForPreTraining,
            RoCBertForQuestionAnswering,
            RoCBertForSequenceClassification,
            RoCBertForTokenClassification,
            RoCBertModel,
            RoCBertPreTrainedModel,
            load_tf_weights_in_roc_bert,
        )
        from transformers.src.transformers.models.roformer import (
            RoFormerForCausalLM,
            RoFormerForMaskedLM,
            RoFormerForMultipleChoice,
            RoFormerForQuestionAnswering,
            RoFormerForSequenceClassification,
            RoFormerForTokenClassification,
            RoFormerModel,
            RoFormerPreTrainedModel,
            load_tf_weights_in_roformer,
        )
        from transformers.src.transformers.models.rt_detr import (
            RTDetrForObjectDetection,
            RTDetrModel,
            RTDetrPreTrainedModel,
            RTDetrResNetBackbone,
            RTDetrResNetPreTrainedModel,
        )
        from transformers.src.transformers.models.rwkv import (
            RwkvForCausalLM,
            RwkvModel,
            RwkvPreTrainedModel,
        )
        from transformers.src.transformers.models.sam import (
            SamModel,
            SamPreTrainedModel,
        )
        from transformers.src.transformers.models.seamless_m4t import (
            SeamlessM4TCodeHifiGan,
            SeamlessM4TForSpeechToSpeech,
            SeamlessM4TForSpeechToText,
            SeamlessM4TForTextToSpeech,
            SeamlessM4TForTextToText,
            SeamlessM4THifiGan,
            SeamlessM4TModel,
            SeamlessM4TPreTrainedModel,
            SeamlessM4TTextToUnitForConditionalGeneration,
            SeamlessM4TTextToUnitModel,
        )
        from transformers.src.transformers.models.seamless_m4t_v2 import (
            SeamlessM4Tv2ForSpeechToSpeech,
            SeamlessM4Tv2ForSpeechToText,
            SeamlessM4Tv2ForTextToSpeech,
            SeamlessM4Tv2ForTextToText,
            SeamlessM4Tv2Model,
            SeamlessM4Tv2PreTrainedModel,
        )
        from transformers.src.transformers.models.segformer import (
            SegformerDecodeHead,
            SegformerForImageClassification,
            SegformerForSemanticSegmentation,
            SegformerModel,
            SegformerPreTrainedModel,
        )
        from transformers.src.transformers.models.seggpt import (
            SegGptForImageSegmentation,
            SegGptModel,
            SegGptPreTrainedModel,
        )
        from transformers.src.transformers.models.sew import (
            SEWForCTC,
            SEWForSequenceClassification,
            SEWModel,
            SEWPreTrainedModel,
        )
        from transformers.src.transformers.models.sew_d import (
            SEWDForCTC,
            SEWDForSequenceClassification,
            SEWDModel,
            SEWDPreTrainedModel,
        )
        from transformers.src.transformers.models.siglip import (
            SiglipForImageClassification,
            SiglipModel,
            SiglipPreTrainedModel,
            SiglipTextModel,
            SiglipVisionModel,
        )
        from transformers.src.transformers.models.speech_encoder_decoder import SpeechEncoderDecoderModel
        from transformers.src.transformers.models.speech_to_text import (
            Speech2TextForConditionalGeneration,
            Speech2TextModel,
            Speech2TextPreTrainedModel,
        )
        from transformers.src.transformers.models.speecht5 import (
            SpeechT5ForSpeechToSpeech,
            SpeechT5ForSpeechToText,
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Model,
            SpeechT5PreTrainedModel,
        )
        from transformers.src.transformers.models.splinter import (
            SplinterForPreTraining,
            SplinterForQuestionAnswering,
            SplinterModel,
            SplinterPreTrainedModel,
        )
        from transformers.src.transformers.models.squeezebert import (
            SqueezeBertForMaskedLM,
            SqueezeBertForMultipleChoice,
            SqueezeBertForQuestionAnswering,
            SqueezeBertForSequenceClassification,
            SqueezeBertForTokenClassification,
            SqueezeBertModel,
            SqueezeBertPreTrainedModel,
        )
        from transformers.src.transformers.models.stablelm import (
            StableLmForCausalLM,
            StableLmForSequenceClassification,
            StableLmForTokenClassification,
            StableLmModel,
            StableLmPreTrainedModel,
        )
        from transformers.src.transformers.models.starcoder2 import (
            Starcoder2ForCausalLM,
            Starcoder2ForSequenceClassification,
            Starcoder2ForTokenClassification,
            Starcoder2Model,
            Starcoder2PreTrainedModel,
        )
        from transformers.src.transformers.models.superpoint import (
            SuperPointForKeypointDetection,
            SuperPointPreTrainedModel,
        )
        from transformers.src.transformers.models.swiftformer import (
            SwiftFormerForImageClassification,
            SwiftFormerModel,
            SwiftFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.swin import (
            SwinBackbone,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            SwinPreTrainedModel,
        )
        from transformers.src.transformers.models.swin2sr import (
            Swin2SRForImageSuperResolution,
            Swin2SRModel,
            Swin2SRPreTrainedModel,
        )
        from transformers.src.transformers.models.swinv2 import (
            Swinv2Backbone,
            Swinv2ForImageClassification,
            Swinv2ForMaskedImageModeling,
            Swinv2Model,
            Swinv2PreTrainedModel,
        )
        from transformers.src.transformers.models.switch_transformers import (
            SwitchTransformersEncoderModel,
            SwitchTransformersForConditionalGeneration,
            SwitchTransformersModel,
            SwitchTransformersPreTrainedModel,
            SwitchTransformersSparseMLP,
            SwitchTransformersTop1Router,
        )
        from transformers.src.transformers.models.t5 import (
            T5EncoderModel,
            T5ForConditionalGeneration,
            T5ForQuestionAnswering,
            T5ForSequenceClassification,
            T5ForTokenClassification,
            T5Model,
            T5PreTrainedModel,
            load_tf_weights_in_t5,
        )
        from transformers.src.transformers.models.table_transformer import (
            TableTransformerForObjectDetection,
            TableTransformerModel,
            TableTransformerPreTrainedModel,
        )
        from transformers.src.transformers.models.tapas import (
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
            TapasModel,
            TapasPreTrainedModel,
            load_tf_weights_in_tapas,
        )
        from transformers.src.transformers.models.time_series_transformer import (
            TimeSeriesTransformerForPrediction,
            TimeSeriesTransformerModel,
            TimeSeriesTransformerPreTrainedModel,
        )
        from transformers.src.transformers.models.timesformer import (
            TimesformerForVideoClassification,
            TimesformerModel,
            TimesformerPreTrainedModel,
        )
        from transformers.src.transformers.models.timm_backbone import TimmBackbone
        from transformers.src.transformers.models.trocr import (
            TrOCRForCausalLM,
            TrOCRPreTrainedModel,
        )
        from transformers.src.transformers.models.tvp import (
            TvpForVideoGrounding,
            TvpModel,
            TvpPreTrainedModel,
        )
        from transformers.src.transformers.models.udop import (
            UdopEncoderModel,
            UdopForConditionalGeneration,
            UdopModel,
            UdopPreTrainedModel,
        )
        from transformers.src.transformers.models.umt5 import (
            UMT5EncoderModel,
            UMT5ForConditionalGeneration,
            UMT5ForQuestionAnswering,
            UMT5ForSequenceClassification,
            UMT5ForTokenClassification,
            UMT5Model,
            UMT5PreTrainedModel,
        )
        from transformers.src.transformers.models.unispeech import (
            UniSpeechForCTC,
            UniSpeechForPreTraining,
            UniSpeechForSequenceClassification,
            UniSpeechModel,
            UniSpeechPreTrainedModel,
        )
        from transformers.src.transformers.models.unispeech_sat import (
            UniSpeechSatForAudioFrameClassification,
            UniSpeechSatForCTC,
            UniSpeechSatForPreTraining,
            UniSpeechSatForSequenceClassification,
            UniSpeechSatForXVector,
            UniSpeechSatModel,
            UniSpeechSatPreTrainedModel,
        )
        from transformers.src.transformers.models.univnet import UnivNetModel
        from transformers.src.transformers.models.upernet import (
            UperNetForSemanticSegmentation,
            UperNetPreTrainedModel,
        )
        from transformers.src.transformers.models.video_llava import (
            VideoLlavaForConditionalGeneration,
            VideoLlavaPreTrainedModel,
            VideoLlavaProcessor,
        )
        from transformers.src.transformers.models.videomae import (
            VideoMAEForPreTraining,
            VideoMAEForVideoClassification,
            VideoMAEModel,
            VideoMAEPreTrainedModel,
        )
        from transformers.src.transformers.models.vilt import (
            ViltForImageAndTextRetrieval,
            ViltForImagesAndTextClassification,
            ViltForMaskedLM,
            ViltForQuestionAnswering,
            ViltForTokenClassification,
            ViltModel,
            ViltPreTrainedModel,
        )
        from transformers.src.transformers.models.vipllava import (
            VipLlavaForConditionalGeneration,
            VipLlavaPreTrainedModel,
        )
        from transformers.src.transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel
        from transformers.src.transformers.models.vision_text_dual_encoder import VisionTextDualEncoderModel
        from transformers.src.transformers.models.visual_bert import (
            VisualBertForMultipleChoice,
            VisualBertForPreTraining,
            VisualBertForQuestionAnswering,
            VisualBertForRegionToPhraseAlignment,
            VisualBertForVisualReasoning,
            VisualBertModel,
            VisualBertPreTrainedModel,
        )
        from transformers.src.transformers.models.vit import (
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )
        from transformers.src.transformers.models.vit_mae import (
            ViTMAEForPreTraining,
            ViTMAEModel,
            ViTMAEPreTrainedModel,
        )
        from transformers.src.transformers.models.vit_msn import (
            ViTMSNForImageClassification,
            ViTMSNModel,
            ViTMSNPreTrainedModel,
        )
        from transformers.src.transformers.models.vitdet import (
            VitDetBackbone,
            VitDetModel,
            VitDetPreTrainedModel,
        )
        from transformers.src.transformers.models.vitmatte import (
            VitMatteForImageMatting,
            VitMattePreTrainedModel,
        )
        from transformers.src.transformers.models.vits import (
            VitsModel,
            VitsPreTrainedModel,
        )
        from transformers.src.transformers.models.vivit import (
            VivitForVideoClassification,
            VivitModel,
            VivitPreTrainedModel,
        )
        from transformers.src.transformers.models.wav2vec2 import (
            Wav2Vec2ForAudioFrameClassification,
            Wav2Vec2ForCTC,
            Wav2Vec2ForMaskedLM,
            Wav2Vec2ForPreTraining,
            Wav2Vec2ForSequenceClassification,
            Wav2Vec2ForXVector,
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
        from transformers.src.transformers.models.wav2vec2_bert import (
            Wav2Vec2BertForAudioFrameClassification,
            Wav2Vec2BertForCTC,
            Wav2Vec2BertForSequenceClassification,
            Wav2Vec2BertForXVector,
            Wav2Vec2BertModel,
            Wav2Vec2BertPreTrainedModel,
        )
        from transformers.src.transformers.models.wav2vec2_conformer import (
            Wav2Vec2ConformerForAudioFrameClassification,
            Wav2Vec2ConformerForCTC,
            Wav2Vec2ConformerForPreTraining,
            Wav2Vec2ConformerForSequenceClassification,
            Wav2Vec2ConformerForXVector,
            Wav2Vec2ConformerModel,
            Wav2Vec2ConformerPreTrainedModel,
        )
        from transformers.src.transformers.models.wavlm import (
            WavLMForAudioFrameClassification,
            WavLMForCTC,
            WavLMForSequenceClassification,
            WavLMForXVector,
            WavLMModel,
            WavLMPreTrainedModel,
        )
        from transformers.src.transformers.models.whisper import (
            WhisperForAudioClassification,
            WhisperForCausalLM,
            WhisperForConditionalGeneration,
            WhisperModel,
            WhisperPreTrainedModel,
        )
        from transformers.src.transformers.models.x_clip import (
            XCLIPModel,
            XCLIPPreTrainedModel,
            XCLIPTextModel,
            XCLIPVisionModel,
        )
        from transformers.src.transformers.models.xglm import (
            XGLMForCausalLM,
            XGLMModel,
            XGLMPreTrainedModel,
        )
        from transformers.src.transformers.models.xlm import (
            XLMForMultipleChoice,
            XLMForQuestionAnswering,
            XLMForQuestionAnsweringSimple,
            XLMForSequenceClassification,
            XLMForTokenClassification,
            XLMModel,
            XLMPreTrainedModel,
            XLMWithLMHeadModel,
        )
        from transformers.src.transformers.models.xlm_roberta import (
            XLMRobertaForCausalLM,
            XLMRobertaForMaskedLM,
            XLMRobertaForMultipleChoice,
            XLMRobertaForQuestionAnswering,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaModel,
            XLMRobertaPreTrainedModel,
        )
        from transformers.src.transformers.models.xlm_roberta_xl import (
            XLMRobertaXLForCausalLM,
            XLMRobertaXLForMaskedLM,
            XLMRobertaXLForMultipleChoice,
            XLMRobertaXLForQuestionAnswering,
            XLMRobertaXLForSequenceClassification,
            XLMRobertaXLForTokenClassification,
            XLMRobertaXLModel,
            XLMRobertaXLPreTrainedModel,
        )
        from transformers.src.transformers.models.xlnet import (
            XLNetForMultipleChoice,
            XLNetForQuestionAnswering,
            XLNetForQuestionAnsweringSimple,
            XLNetForSequenceClassification,
            XLNetForTokenClassification,
            XLNetLMHeadModel,
            XLNetModel,
            XLNetPreTrainedModel,
            load_tf_weights_in_xlnet,
        )
        from transformers.src.transformers.models.xmod import (
            XmodForCausalLM,
            XmodForMaskedLM,
            XmodForMultipleChoice,
            XmodForQuestionAnswering,
            XmodForSequenceClassification,
            XmodForTokenClassification,
            XmodModel,
            XmodPreTrainedModel,
        )
        from transformers.src.transformers.models.yolos import (
            YolosForObjectDetection,
            YolosModel,
            YolosPreTrainedModel,
        )
        from transformers.src.transformers.models.yoso import (
            YosoForMaskedLM,
            YosoForMultipleChoice,
            YosoForQuestionAnswering,
            YosoForSequenceClassification,
            YosoForTokenClassification,
            YosoModel,
            YosoPreTrainedModel,
        )
        from transformers.src.transformers.models.zamba import (
            ZambaForCausalLM,
            ZambaForSequenceClassification,
            ZambaModel,
            ZambaPreTrainedModel,
        )
        from transformers.src.transformers.models.zoedepth import (
            ZoeDepthForDepthEstimation,
            ZoeDepthPreTrainedModel,
        )

        # Optimization
        from transformers.src.transformers.optimization import (
            Adafactor,
            AdamW,
            get_constant_schedule,
            get_constant_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
            get_cosine_with_hard_restarts_schedule_with_warmup,
            get_inverse_sqrt_schedule,
            get_linear_schedule_with_warmup,
            get_polynomial_decay_schedule_with_warmup,
            get_scheduler,
            get_wsd_schedule,
        )
        from transformers.src.transformers.pytorch_utils import Conv1D, apply_chunking_to_forward, prune_layer

        # Trainer
        from transformers.src.transformers.trainer import Trainer
        from transformers.src.transformers.trainer_pt_utils import torch_distributed_zero_first
        from transformers.src.transformers.trainer_seq2seq import Seq2SeqTrainer

    # TensorFlow
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from transformers.src.transformers.utils.dummy_tf_objects import *
    else:
        from transformers.src.transformers.benchmark.benchmark_args_tf import TensorFlowBenchmarkArguments

        # Benchmarks
        from transformers.src.transformers.benchmark.benchmark_tf import TensorFlowBenchmark
        from transformers.src.transformers.generation import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
            TFGenerationMixin,
            TFLogitsProcessor,
            TFLogitsProcessorList,
            TFLogitsWarper,
            TFMinLengthLogitsProcessor,
            TFNoBadWordsLogitsProcessor,
            TFNoRepeatNGramLogitsProcessor,
            TFRepetitionPenaltyLogitsProcessor,
            TFSuppressTokensAtBeginLogitsProcessor,
            TFSuppressTokensLogitsProcessor,
            TFTemperatureLogitsWarper,
            TFTopKLogitsWarper,
            TFTopPLogitsWarper,
        )
        from transformers.src.transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
        from transformers.src.transformers.modeling_tf_utils import (
            TFPreTrainedModel,
            TFSequenceSummary,
            TFSharedEmbeddings,
            shape_list,
        )

        # TensorFlow model imports
        from transformers.src.transformers.models.albert import (
            TFAlbertForMaskedLM,
            TFAlbertForMultipleChoice,
            TFAlbertForPreTraining,
            TFAlbertForQuestionAnswering,
            TFAlbertForSequenceClassification,
            TFAlbertForTokenClassification,
            TFAlbertMainLayer,
            TFAlbertModel,
            TFAlbertPreTrainedModel,
        )
        from transformers.src.transformers.models.auto import (
            TF_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_MASK_GENERATION_MAPPING,
            TF_MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            TF_MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_TEXT_ENCODING_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING,
            TF_MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForAudioClassification,
            TFAutoModelForCausalLM,
            TFAutoModelForDocumentQuestionAnswering,
            TFAutoModelForImageClassification,
            TFAutoModelForMaskedImageModeling,
            TFAutoModelForMaskedLM,
            TFAutoModelForMaskGeneration,
            TFAutoModelForMultipleChoice,
            TFAutoModelForNextSentencePrediction,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSemanticSegmentation,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForSpeechSeq2Seq,
            TFAutoModelForTableQuestionAnswering,
            TFAutoModelForTextEncoding,
            TFAutoModelForTokenClassification,
            TFAutoModelForVision2Seq,
            TFAutoModelForZeroShotImageClassification,
            TFAutoModelWithLMHead,
        )
        from transformers.src.transformers.models.bart import (
            TFBartForConditionalGeneration,
            TFBartForSequenceClassification,
            TFBartModel,
            TFBartPretrainedModel,
        )
        from transformers.src.transformers.models.bert import (
            TFBertForMaskedLM,
            TFBertForMultipleChoice,
            TFBertForNextSentencePrediction,
            TFBertForPreTraining,
            TFBertForQuestionAnswering,
            TFBertForSequenceClassification,
            TFBertForTokenClassification,
            TFBertLMHeadModel,
            TFBertMainLayer,
            TFBertModel,
            TFBertPreTrainedModel,
        )
        from transformers.src.transformers.models.blenderbot import (
            TFBlenderbotForConditionalGeneration,
            TFBlenderbotModel,
            TFBlenderbotPreTrainedModel,
        )
        from transformers.src.transformers.models.blenderbot_small import (
            TFBlenderbotSmallForConditionalGeneration,
            TFBlenderbotSmallModel,
            TFBlenderbotSmallPreTrainedModel,
        )
        from transformers.src.transformers.models.blip import (
            TFBlipForConditionalGeneration,
            TFBlipForImageTextRetrieval,
            TFBlipForQuestionAnswering,
            TFBlipModel,
            TFBlipPreTrainedModel,
            TFBlipTextModel,
            TFBlipVisionModel,
        )
        from transformers.src.transformers.models.camembert import (
            TFCamembertForCausalLM,
            TFCamembertForMaskedLM,
            TFCamembertForMultipleChoice,
            TFCamembertForQuestionAnswering,
            TFCamembertForSequenceClassification,
            TFCamembertForTokenClassification,
            TFCamembertModel,
            TFCamembertPreTrainedModel,
        )
        from transformers.src.transformers.models.clip import (
            TFCLIPModel,
            TFCLIPPreTrainedModel,
            TFCLIPTextModel,
            TFCLIPVisionModel,
        )
        from transformers.src.transformers.models.convbert import (
            TFConvBertForMaskedLM,
            TFConvBertForMultipleChoice,
            TFConvBertForQuestionAnswering,
            TFConvBertForSequenceClassification,
            TFConvBertForTokenClassification,
            TFConvBertModel,
            TFConvBertPreTrainedModel,
        )
        from transformers.src.transformers.models.convnext import (
            TFConvNextForImageClassification,
            TFConvNextModel,
            TFConvNextPreTrainedModel,
        )
        from transformers.src.transformers.models.convnextv2 import (
            TFConvNextV2ForImageClassification,
            TFConvNextV2Model,
            TFConvNextV2PreTrainedModel,
        )
        from transformers.src.transformers.models.ctrl import (
            TFCTRLForSequenceClassification,
            TFCTRLLMHeadModel,
            TFCTRLModel,
            TFCTRLPreTrainedModel,
        )
        from transformers.src.transformers.models.cvt import (
            TFCvtForImageClassification,
            TFCvtModel,
            TFCvtPreTrainedModel,
        )
        from transformers.src.transformers.models.data2vec import (
            TFData2VecVisionForImageClassification,
            TFData2VecVisionForSemanticSegmentation,
            TFData2VecVisionModel,
            TFData2VecVisionPreTrainedModel,
        )
        from transformers.src.transformers.models.deberta import (
            TFDebertaForMaskedLM,
            TFDebertaForQuestionAnswering,
            TFDebertaForSequenceClassification,
            TFDebertaForTokenClassification,
            TFDebertaModel,
            TFDebertaPreTrainedModel,
        )
        from transformers.src.transformers.models.deberta_v2 import (
            TFDebertaV2ForMaskedLM,
            TFDebertaV2ForMultipleChoice,
            TFDebertaV2ForQuestionAnswering,
            TFDebertaV2ForSequenceClassification,
            TFDebertaV2ForTokenClassification,
            TFDebertaV2Model,
            TFDebertaV2PreTrainedModel,
        )
        from transformers.src.transformers.models.deit import (
            TFDeiTForImageClassification,
            TFDeiTForImageClassificationWithTeacher,
            TFDeiTForMaskedImageModeling,
            TFDeiTModel,
            TFDeiTPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.efficientformer import (
            TFEfficientFormerForImageClassification,
            TFEfficientFormerForImageClassificationWithTeacher,
            TFEfficientFormerModel,
            TFEfficientFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.deprecated.transfo_xl import (
            TFAdaptiveEmbedding,
            TFTransfoXLForSequenceClassification,
            TFTransfoXLLMHeadModel,
            TFTransfoXLMainLayer,
            TFTransfoXLModel,
            TFTransfoXLPreTrainedModel,
        )
        from transformers.src.transformers.models.distilbert import (
            TFDistilBertForMaskedLM,
            TFDistilBertForMultipleChoice,
            TFDistilBertForQuestionAnswering,
            TFDistilBertForSequenceClassification,
            TFDistilBertForTokenClassification,
            TFDistilBertMainLayer,
            TFDistilBertModel,
            TFDistilBertPreTrainedModel,
        )
        from transformers.src.transformers.models.dpr import (
            TFDPRContextEncoder,
            TFDPRPretrainedContextEncoder,
            TFDPRPretrainedQuestionEncoder,
            TFDPRPretrainedReader,
            TFDPRQuestionEncoder,
            TFDPRReader,
        )
        from transformers.src.transformers.models.electra import (
            TFElectraForMaskedLM,
            TFElectraForMultipleChoice,
            TFElectraForPreTraining,
            TFElectraForQuestionAnswering,
            TFElectraForSequenceClassification,
            TFElectraForTokenClassification,
            TFElectraModel,
            TFElectraPreTrainedModel,
        )
        from transformers.src.transformers.models.encoder_decoder import TFEncoderDecoderModel
        from transformers.src.transformers.models.esm import (
            TFEsmForMaskedLM,
            TFEsmForSequenceClassification,
            TFEsmForTokenClassification,
            TFEsmModel,
            TFEsmPreTrainedModel,
        )
        from transformers.src.transformers.models.flaubert import (
            TFFlaubertForMultipleChoice,
            TFFlaubertForQuestionAnsweringSimple,
            TFFlaubertForSequenceClassification,
            TFFlaubertForTokenClassification,
            TFFlaubertModel,
            TFFlaubertPreTrainedModel,
            TFFlaubertWithLMHeadModel,
        )
        from transformers.src.transformers.models.funnel import (
            TFFunnelBaseModel,
            TFFunnelForMaskedLM,
            TFFunnelForMultipleChoice,
            TFFunnelForPreTraining,
            TFFunnelForQuestionAnswering,
            TFFunnelForSequenceClassification,
            TFFunnelForTokenClassification,
            TFFunnelModel,
            TFFunnelPreTrainedModel,
        )
        from transformers.src.transformers.models.gpt2 import (
            TFGPT2DoubleHeadsModel,
            TFGPT2ForSequenceClassification,
            TFGPT2LMHeadModel,
            TFGPT2MainLayer,
            TFGPT2Model,
            TFGPT2PreTrainedModel,
        )
        from transformers.src.transformers.models.gptj import (
            TFGPTJForCausalLM,
            TFGPTJForQuestionAnswering,
            TFGPTJForSequenceClassification,
            TFGPTJModel,
            TFGPTJPreTrainedModel,
        )
        from transformers.src.transformers.models.groupvit import (
            TFGroupViTModel,
            TFGroupViTPreTrainedModel,
            TFGroupViTTextModel,
            TFGroupViTVisionModel,
        )
        from transformers.src.transformers.models.hubert import (
            TFHubertForCTC,
            TFHubertModel,
            TFHubertPreTrainedModel,
        )
        from transformers.src.transformers.models.idefics import (
            TFIdeficsForVisionText2Text,
            TFIdeficsModel,
            TFIdeficsPreTrainedModel,
        )
        from transformers.src.transformers.models.layoutlm import (
            TFLayoutLMForMaskedLM,
            TFLayoutLMForQuestionAnswering,
            TFLayoutLMForSequenceClassification,
            TFLayoutLMForTokenClassification,
            TFLayoutLMMainLayer,
            TFLayoutLMModel,
            TFLayoutLMPreTrainedModel,
        )
        from transformers.src.transformers.models.layoutlmv3 import (
            TFLayoutLMv3ForQuestionAnswering,
            TFLayoutLMv3ForSequenceClassification,
            TFLayoutLMv3ForTokenClassification,
            TFLayoutLMv3Model,
            TFLayoutLMv3PreTrainedModel,
        )
        from transformers.src.transformers.models.led import (
            TFLEDForConditionalGeneration,
            TFLEDModel,
            TFLEDPreTrainedModel,
        )
        from transformers.src.transformers.models.longformer import (
            TFLongformerForMaskedLM,
            TFLongformerForMultipleChoice,
            TFLongformerForQuestionAnswering,
            TFLongformerForSequenceClassification,
            TFLongformerForTokenClassification,
            TFLongformerModel,
            TFLongformerPreTrainedModel,
        )
        from transformers.src.transformers.models.lxmert import (
            TFLxmertForPreTraining,
            TFLxmertMainLayer,
            TFLxmertModel,
            TFLxmertPreTrainedModel,
            TFLxmertVisualFeatureEncoder,
        )
        from transformers.src.transformers.models.marian import (
            TFMarianModel,
            TFMarianMTModel,
            TFMarianPreTrainedModel,
        )
        from transformers.src.transformers.models.mbart import (
            TFMBartForConditionalGeneration,
            TFMBartModel,
            TFMBartPreTrainedModel,
        )
        from transformers.src.transformers.models.mistral import (
            TFMistralForCausalLM,
            TFMistralForSequenceClassification,
            TFMistralModel,
            TFMistralPreTrainedModel,
        )
        from transformers.src.transformers.models.mobilebert import (
            TFMobileBertForMaskedLM,
            TFMobileBertForMultipleChoice,
            TFMobileBertForNextSentencePrediction,
            TFMobileBertForPreTraining,
            TFMobileBertForQuestionAnswering,
            TFMobileBertForSequenceClassification,
            TFMobileBertForTokenClassification,
            TFMobileBertMainLayer,
            TFMobileBertModel,
            TFMobileBertPreTrainedModel,
        )
        from transformers.src.transformers.models.mobilevit import (
            TFMobileViTForImageClassification,
            TFMobileViTForSemanticSegmentation,
            TFMobileViTModel,
            TFMobileViTPreTrainedModel,
        )
        from transformers.src.transformers.models.mpnet import (
            TFMPNetForMaskedLM,
            TFMPNetForMultipleChoice,
            TFMPNetForQuestionAnswering,
            TFMPNetForSequenceClassification,
            TFMPNetForTokenClassification,
            TFMPNetMainLayer,
            TFMPNetModel,
            TFMPNetPreTrainedModel,
        )
        from transformers.src.transformers.models.mt5 import (
            TFMT5EncoderModel,
            TFMT5ForConditionalGeneration,
            TFMT5Model,
        )
        from transformers.src.transformers.models.openai import (
            TFOpenAIGPTDoubleHeadsModel,
            TFOpenAIGPTForSequenceClassification,
            TFOpenAIGPTLMHeadModel,
            TFOpenAIGPTMainLayer,
            TFOpenAIGPTModel,
            TFOpenAIGPTPreTrainedModel,
        )
        from transformers.src.transformers.models.opt import TFOPTForCausalLM, TFOPTModel, TFOPTPreTrainedModel
        from transformers.src.transformers.models.pegasus import (
            TFPegasusForConditionalGeneration,
            TFPegasusModel,
            TFPegasusPreTrainedModel,
        )
        from transformers.src.transformers.models.rag import (
            TFRagModel,
            TFRagPreTrainedModel,
            TFRagSequenceForGeneration,
            TFRagTokenForGeneration,
        )
        from transformers.src.transformers.models.regnet import (
            TFRegNetForImageClassification,
            TFRegNetModel,
            TFRegNetPreTrainedModel,
        )
        from transformers.src.transformers.models.rembert import (
            TFRemBertForCausalLM,
            TFRemBertForMaskedLM,
            TFRemBertForMultipleChoice,
            TFRemBertForQuestionAnswering,
            TFRemBertForSequenceClassification,
            TFRemBertForTokenClassification,
            TFRemBertModel,
            TFRemBertPreTrainedModel,
        )
        from transformers.src.transformers.models.resnet import (
            TFResNetForImageClassification,
            TFResNetModel,
            TFResNetPreTrainedModel,
        )
        from transformers.src.transformers.models.roberta import (
            TFRobertaForCausalLM,
            TFRobertaForMaskedLM,
            TFRobertaForMultipleChoice,
            TFRobertaForQuestionAnswering,
            TFRobertaForSequenceClassification,
            TFRobertaForTokenClassification,
            TFRobertaMainLayer,
            TFRobertaModel,
            TFRobertaPreTrainedModel,
        )
        from transformers.src.transformers.models.roberta_prelayernorm import (
            TFRobertaPreLayerNormForCausalLM,
            TFRobertaPreLayerNormForMaskedLM,
            TFRobertaPreLayerNormForMultipleChoice,
            TFRobertaPreLayerNormForQuestionAnswering,
            TFRobertaPreLayerNormForSequenceClassification,
            TFRobertaPreLayerNormForTokenClassification,
            TFRobertaPreLayerNormMainLayer,
            TFRobertaPreLayerNormModel,
            TFRobertaPreLayerNormPreTrainedModel,
        )
        from transformers.src.transformers.models.roformer import (
            TFRoFormerForCausalLM,
            TFRoFormerForMaskedLM,
            TFRoFormerForMultipleChoice,
            TFRoFormerForQuestionAnswering,
            TFRoFormerForSequenceClassification,
            TFRoFormerForTokenClassification,
            TFRoFormerModel,
            TFRoFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.sam import (
            TFSamModel,
            TFSamPreTrainedModel,
        )
        from transformers.src.transformers.models.segformer import (
            TFSegformerDecodeHead,
            TFSegformerForImageClassification,
            TFSegformerForSemanticSegmentation,
            TFSegformerModel,
            TFSegformerPreTrainedModel,
        )
        from transformers.src.transformers.models.speech_to_text import (
            TFSpeech2TextForConditionalGeneration,
            TFSpeech2TextModel,
            TFSpeech2TextPreTrainedModel,
        )
        from transformers.src.transformers.models.swiftformer import (
            TFSwiftFormerForImageClassification,
            TFSwiftFormerModel,
            TFSwiftFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.swin import (
            TFSwinForImageClassification,
            TFSwinForMaskedImageModeling,
            TFSwinModel,
            TFSwinPreTrainedModel,
        )
        from transformers.src.transformers.models.t5 import (
            TFT5EncoderModel,
            TFT5ForConditionalGeneration,
            TFT5Model,
            TFT5PreTrainedModel,
        )
        from transformers.src.transformers.models.tapas import (
            TFTapasForMaskedLM,
            TFTapasForQuestionAnswering,
            TFTapasForSequenceClassification,
            TFTapasModel,
            TFTapasPreTrainedModel,
        )
        from transformers.src.transformers.models.vision_encoder_decoder import TFVisionEncoderDecoderModel
        from transformers.src.transformers.models.vision_text_dual_encoder import TFVisionTextDualEncoderModel
        from transformers.src.transformers.models.vit import (
            TFViTForImageClassification,
            TFViTModel,
            TFViTPreTrainedModel,
        )
        from transformers.src.transformers.models.vit_mae import (
            TFViTMAEForPreTraining,
            TFViTMAEModel,
            TFViTMAEPreTrainedModel,
        )
        from transformers.src.transformers.models.wav2vec2 import (
            TFWav2Vec2ForCTC,
            TFWav2Vec2ForSequenceClassification,
            TFWav2Vec2Model,
            TFWav2Vec2PreTrainedModel,
        )
        from transformers.src.transformers.models.whisper import (
            TFWhisperForConditionalGeneration,
            TFWhisperModel,
            TFWhisperPreTrainedModel,
        )
        from transformers.src.transformers.models.xglm import (
            TFXGLMForCausalLM,
            TFXGLMModel,
            TFXGLMPreTrainedModel,
        )
        from transformers.src.transformers.models.xlm import (
            TFXLMForMultipleChoice,
            TFXLMForQuestionAnsweringSimple,
            TFXLMForSequenceClassification,
            TFXLMForTokenClassification,
            TFXLMMainLayer,
            TFXLMModel,
            TFXLMPreTrainedModel,
            TFXLMWithLMHeadModel,
        )
        from transformers.src.transformers.models.xlm_roberta import (
            TFXLMRobertaForCausalLM,
            TFXLMRobertaForMaskedLM,
            TFXLMRobertaForMultipleChoice,
            TFXLMRobertaForQuestionAnswering,
            TFXLMRobertaForSequenceClassification,
            TFXLMRobertaForTokenClassification,
            TFXLMRobertaModel,
            TFXLMRobertaPreTrainedModel,
        )
        from transformers.src.transformers.models.xlnet import (
            TFXLNetForMultipleChoice,
            TFXLNetForQuestionAnsweringSimple,
            TFXLNetForSequenceClassification,
            TFXLNetForTokenClassification,
            TFXLNetLMHeadModel,
            TFXLNetMainLayer,
            TFXLNetModel,
            TFXLNetPreTrainedModel,
        )

        # Optimization
        from transformers.src.transformers.optimization_tf import (
            AdamWeightDecay,
            GradientAccumulator,
            WarmUp,
            create_optimizer,
        )

    try:
        if not (
            is_librosa_available()
            and is_essentia_available()
            and is_scipy_available()
            and is_torch_available()
            and is_pretty_midi_available()
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_essentia_and_librosa_and_pretty_midi_and_scipy_and_torch_objects import *
    else:
        from transformers.src.transformers.models.pop2piano import (
            Pop2PianoFeatureExtractor,
            Pop2PianoProcessor,
            Pop2PianoTokenizer,
        )

    try:
        if not is_torchaudio_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from transformers.src.transformers.utils.dummy_torchaudio_objects import *
    else:
        from transformers.src.transformers.models.musicgen_melody import MusicgenMelodyFeatureExtractor, MusicgenMelodyProcessor
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # Import the same objects as dummies to get them in the namespace.
        # They will raise an import error if the user tries to instantiate / use them.
        from transformers.src.transformers.utils.dummy_flax_objects import *
    else:
        from transformers.src.transformers.generation import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxForceTokensLogitsProcessor,
            FlaxGenerationMixin,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxSuppressTokensAtBeginLogitsProcessor,
            FlaxSuppressTokensLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
            FlaxWhisperTimeStampLogitsProcessor,
        )
        from transformers.src.transformers.modeling_flax_utils import FlaxPreTrainedModel

        # Flax model imports
        from transformers.src.transformers.models.albert import (
            FlaxAlbertForMaskedLM,
            FlaxAlbertForMultipleChoice,
            FlaxAlbertForPreTraining,
            FlaxAlbertForQuestionAnswering,
            FlaxAlbertForSequenceClassification,
            FlaxAlbertForTokenClassification,
            FlaxAlbertModel,
            FlaxAlbertPreTrainedModel,
        )
        from transformers.src.transformers.models.auto import (
            FLAX_MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_MASKED_LM_MAPPING,
            FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            FLAX_MODEL_FOR_PRETRAINING_MAPPING,
            FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING,
            FLAX_MODEL_MAPPING,
            FlaxAutoModel,
            FlaxAutoModelForCausalLM,
            FlaxAutoModelForImageClassification,
            FlaxAutoModelForMaskedLM,
            FlaxAutoModelForMultipleChoice,
            FlaxAutoModelForNextSentencePrediction,
            FlaxAutoModelForPreTraining,
            FlaxAutoModelForQuestionAnswering,
            FlaxAutoModelForSeq2SeqLM,
            FlaxAutoModelForSequenceClassification,
            FlaxAutoModelForSpeechSeq2Seq,
            FlaxAutoModelForTokenClassification,
            FlaxAutoModelForVision2Seq,
        )
        from transformers.src.transformers.models.bart import (
            FlaxBartDecoderPreTrainedModel,
            FlaxBartForCausalLM,
            FlaxBartForConditionalGeneration,
            FlaxBartForQuestionAnswering,
            FlaxBartForSequenceClassification,
            FlaxBartModel,
            FlaxBartPreTrainedModel,
        )
        from transformers.src.transformers.models.beit import (
            FlaxBeitForImageClassification,
            FlaxBeitForMaskedImageModeling,
            FlaxBeitModel,
            FlaxBeitPreTrainedModel,
        )
        from transformers.src.transformers.models.bert import (
            FlaxBertForCausalLM,
            FlaxBertForMaskedLM,
            FlaxBertForMultipleChoice,
            FlaxBertForNextSentencePrediction,
            FlaxBertForPreTraining,
            FlaxBertForQuestionAnswering,
            FlaxBertForSequenceClassification,
            FlaxBertForTokenClassification,
            FlaxBertModel,
            FlaxBertPreTrainedModel,
        )
        from transformers.src.transformers.models.big_bird import (
            FlaxBigBirdForCausalLM,
            FlaxBigBirdForMaskedLM,
            FlaxBigBirdForMultipleChoice,
            FlaxBigBirdForPreTraining,
            FlaxBigBirdForQuestionAnswering,
            FlaxBigBirdForSequenceClassification,
            FlaxBigBirdForTokenClassification,
            FlaxBigBirdModel,
            FlaxBigBirdPreTrainedModel,
        )
        from transformers.src.transformers.models.blenderbot import (
            FlaxBlenderbotForConditionalGeneration,
            FlaxBlenderbotModel,
            FlaxBlenderbotPreTrainedModel,
        )
        from transformers.src.transformers.models.blenderbot_small import (
            FlaxBlenderbotSmallForConditionalGeneration,
            FlaxBlenderbotSmallModel,
            FlaxBlenderbotSmallPreTrainedModel,
        )
        from transformers.src.transformers.models.bloom import (
            FlaxBloomForCausalLM,
            FlaxBloomModel,
            FlaxBloomPreTrainedModel,
        )
        from transformers.src.transformers.models.clip import (
            FlaxCLIPModel,
            FlaxCLIPPreTrainedModel,
            FlaxCLIPTextModel,
            FlaxCLIPTextModelWithProjection,
            FlaxCLIPTextPreTrainedModel,
            FlaxCLIPVisionModel,
            FlaxCLIPVisionPreTrainedModel,
        )
        from transformers.src.transformers.models.dinov2 import (
            FlaxDinov2ForImageClassification,
            FlaxDinov2Model,
            FlaxDinov2PreTrainedModel,
        )
        from transformers.src.transformers.models.distilbert import (
            FlaxDistilBertForMaskedLM,
            FlaxDistilBertForMultipleChoice,
            FlaxDistilBertForQuestionAnswering,
            FlaxDistilBertForSequenceClassification,
            FlaxDistilBertForTokenClassification,
            FlaxDistilBertModel,
            FlaxDistilBertPreTrainedModel,
        )
        from transformers.src.transformers.models.electra import (
            FlaxElectraForCausalLM,
            FlaxElectraForMaskedLM,
            FlaxElectraForMultipleChoice,
            FlaxElectraForPreTraining,
            FlaxElectraForQuestionAnswering,
            FlaxElectraForSequenceClassification,
            FlaxElectraForTokenClassification,
            FlaxElectraModel,
            FlaxElectraPreTrainedModel,
        )
        from transformers.src.transformers.models.encoder_decoder import FlaxEncoderDecoderModel
        from transformers.src.transformers.models.gemma import (
            FlaxGemmaForCausalLM,
            FlaxGemmaModel,
            FlaxGemmaPreTrainedModel,
        )
        from transformers.src.transformers.models.gpt2 import (
            FlaxGPT2LMHeadModel,
            FlaxGPT2Model,
            FlaxGPT2PreTrainedModel,
        )
        from transformers.src.transformers.models.gpt_neo import (
            FlaxGPTNeoForCausalLM,
            FlaxGPTNeoModel,
            FlaxGPTNeoPreTrainedModel,
        )
        from transformers.src.transformers.models.gptj import (
            FlaxGPTJForCausalLM,
            FlaxGPTJModel,
            FlaxGPTJPreTrainedModel,
        )
        from transformers.src.transformers.models.llama import (
            FlaxLlamaForCausalLM,
            FlaxLlamaModel,
            FlaxLlamaPreTrainedModel,
        )
        from transformers.src.transformers.models.longt5 import (
            FlaxLongT5ForConditionalGeneration,
            FlaxLongT5Model,
            FlaxLongT5PreTrainedModel,
        )
        from transformers.src.transformers.models.marian import (
            FlaxMarianModel,
            FlaxMarianMTModel,
            FlaxMarianPreTrainedModel,
        )
        from transformers.src.transformers.models.mbart import (
            FlaxMBartForConditionalGeneration,
            FlaxMBartForQuestionAnswering,
            FlaxMBartForSequenceClassification,
            FlaxMBartModel,
            FlaxMBartPreTrainedModel,
        )
        from transformers.src.transformers.models.mistral import (
            FlaxMistralForCausalLM,
            FlaxMistralModel,
            FlaxMistralPreTrainedModel,
        )
        from transformers.src.transformers.models.mt5 import (
            FlaxMT5EncoderModel,
            FlaxMT5ForConditionalGeneration,
            FlaxMT5Model,
        )
        from transformers.src.transformers.models.opt import FlaxOPTForCausalLM, FlaxOPTModel, FlaxOPTPreTrainedModel
        from transformers.src.transformers.models.pegasus import (
            FlaxPegasusForConditionalGeneration,
            FlaxPegasusModel,
            FlaxPegasusPreTrainedModel,
        )
        from transformers.src.transformers.models.regnet import (
            FlaxRegNetForImageClassification,
            FlaxRegNetModel,
            FlaxRegNetPreTrainedModel,
        )
        from transformers.src.transformers.models.resnet import (
            FlaxResNetForImageClassification,
            FlaxResNetModel,
            FlaxResNetPreTrainedModel,
        )
        from transformers.src.transformers.models.roberta import (
            FlaxRobertaForCausalLM,
            FlaxRobertaForMaskedLM,
            FlaxRobertaForMultipleChoice,
            FlaxRobertaForQuestionAnswering,
            FlaxRobertaForSequenceClassification,
            FlaxRobertaForTokenClassification,
            FlaxRobertaModel,
            FlaxRobertaPreTrainedModel,
        )
        from transformers.src.transformers.models.roberta_prelayernorm import (
            FlaxRobertaPreLayerNormForCausalLM,
            FlaxRobertaPreLayerNormForMaskedLM,
            FlaxRobertaPreLayerNormForMultipleChoice,
            FlaxRobertaPreLayerNormForQuestionAnswering,
            FlaxRobertaPreLayerNormForSequenceClassification,
            FlaxRobertaPreLayerNormForTokenClassification,
            FlaxRobertaPreLayerNormModel,
            FlaxRobertaPreLayerNormPreTrainedModel,
        )
        from transformers.src.transformers.models.roformer import (
            FlaxRoFormerForMaskedLM,
            FlaxRoFormerForMultipleChoice,
            FlaxRoFormerForQuestionAnswering,
            FlaxRoFormerForSequenceClassification,
            FlaxRoFormerForTokenClassification,
            FlaxRoFormerModel,
            FlaxRoFormerPreTrainedModel,
        )
        from transformers.src.transformers.models.speech_encoder_decoder import FlaxSpeechEncoderDecoderModel
        from transformers.src.transformers.models.t5 import (
            FlaxT5EncoderModel,
            FlaxT5ForConditionalGeneration,
            FlaxT5Model,
            FlaxT5PreTrainedModel,
        )
        from transformers.src.transformers.models.vision_encoder_decoder import FlaxVisionEncoderDecoderModel
        from transformers.src.transformers.models.vision_text_dual_encoder import FlaxVisionTextDualEncoderModel
        from transformers.src.transformers.models.vit import (
            FlaxViTForImageClassification,
            FlaxViTModel,
            FlaxViTPreTrainedModel,
        )
        from transformers.src.transformers.models.wav2vec2 import (
            FlaxWav2Vec2ForCTC,
            FlaxWav2Vec2ForPreTraining,
            FlaxWav2Vec2Model,
            FlaxWav2Vec2PreTrainedModel,
        )
        from transformers.src.transformers.models.whisper import (
            FlaxWhisperForAudioClassification,
            FlaxWhisperForConditionalGeneration,
            FlaxWhisperModel,
            FlaxWhisperPreTrainedModel,
        )
        from transformers.src.transformers.models.xglm import (
            FlaxXGLMForCausalLM,
            FlaxXGLMModel,
            FlaxXGLMPreTrainedModel,
        )
        from transformers.src.transformers.models.xlm_roberta import (
            FlaxXLMRobertaForCausalLM,
            FlaxXLMRobertaForMaskedLM,
            FlaxXLMRobertaForMultipleChoice,
            FlaxXLMRobertaForQuestionAnswering,
            FlaxXLMRobertaForSequenceClassification,
            FlaxXLMRobertaForTokenClassification,
            FlaxXLMRobertaModel,
            FlaxXLMRobertaPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


if not is_tf_available() and not is_torch_available() and not is_flax_available():
    logger.warning_advice(
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. "
        "Models won't be available and only tokenizers, configuration "
        "and file/data utilities can be used."
    )
