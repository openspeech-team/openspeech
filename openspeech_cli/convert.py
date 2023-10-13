from openspeech.encoders.deepspeech2 import DeepSpeech2
import torch


model = DeepSpeech2(
    input_dim=40,
    num_classes=2000,
    rnn_type="lstm",
    num_rnn_layers=0,
    rnn_hidden_dim=256,
)

setattr(model, "forward", model.inference)

new_state_dict = dict()
model_state_dict = torch.load("/home/patrick/prj/openspeech/openspeech_cli/outputs/2023-10-16/11-27-00/torch_model.pt")
for k, v in model_state_dict.items():
    new_state_dict[k.replace("encoder.", "")] = v

model.load_state_dict(new_state_dict)
model.eval()
# import pdb; pdb.set_trace()

dummy_source = torch.rand((1, 201, 40), device="cpu")  # 4s 16k : (1, 401, 40) / 8k : (1, 201, 40)
# dummy_wav_lens = torch.rand((1), device="cpu")
# dummy_inputs = (dummy_source, dummy_wav_lens)

with torch.no_grad():
    torch.onnx.export(
        model,
        args=dummy_source,
        f="/home/patrick/prj/openspeech/openspeech_cli/outputs/2023-10-16/11-27-00/model_vgg.onnx",
        export_params=True,
        input_names=["source"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        keep_initializers_as_inputs=True,
    )
