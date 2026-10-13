import zipfile
import json

old_file = "plant_disease_cnn.keras"
new_file = "plant_disease_cnn_fixed.keras"

with zipfile.ZipFile(old_file, "r") as source:
    config = json.loads(source.read("config.json"))

    # Remove unsupported configuration fields
    for layer in config["config"]["layers"]:
        layer_config = layer.get("config", {})

        if layer_config.get("quantization_config", "missing") is None:
            layer_config.pop("quantization_config", None)

    with zipfile.ZipFile(new_file, "w") as destination:
        for item in source.infolist():
            data = source.read(item.filename)

            if item.filename == "config.json":
                data = json.dumps(config).encode("utf-8")

            destination.writestr(item, data)

print("Model configuration fixed successfully!")