# Copy verified artifacts to shared Drive; no training is started.
import hashlib
import json
import zipfile
from pathlib import Path

import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def artifact_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

handoff_run = Path("/content/drive/MyDrive/SAR/runs/yutao/w1-pure-integrated-v2-001")
handoff_vectors = Path("/content/drive/MyDrive/SAR/releases/stage1/yutao-vectors-001")
handoff_report = json.loads((handoff_run / "run_manifest.json").read_text())
handoff_export = json.loads((handoff_vectors / "manifest.json").read_text())
handoff_checkpoint = Path(handoff_report["checkpoint"])
assert handoff_report["complete"] and handoff_export["training_complete"]
assert handoff_report["run_id"] == handoff_export["run_id"] == "b2ed5728b9a2"
assert artifact_sha(handoff_checkpoint) == handoff_export["checkpoint_sha256"]
handoff_prepared = Path("/content/yutao-pure-integrated-v2")
assert artifact_sha(handoff_prepared / "manifest.json") == handoff_export["binding"]["manifest_sha256"]
handoff_manifest = json.loads((handoff_prepared / "manifest.json").read_text())
handoff_files = {
    "model/" + handoff_checkpoint.name: handoff_checkpoint,
    "model/" + handoff_checkpoint.with_suffix(".complete.json").name: handoff_checkpoint.with_suffix(".complete.json"),
    "model/run_manifest.json": handoff_run / "run_manifest.json",
    "model/config.json": handoff_run / "config.json",
    "vectors/manifest.json": handoff_vectors / "manifest.json",
    "prepared/manifest.json": handoff_prepared / "manifest.json",
    "source/SAR-colab-integrated-v2.zip": Path("/content/drive/MyDrive/SAR-colab-integrated-v2.zip"),
    "source/SAR-colab-integrated-v2.sha256": Path("/content/drive/MyDrive/SAR-colab-integrated-v2.sha256"),
}
for shard in handoff_export["shards"]:
    file = handoff_vectors / shard["path"]
    assert artifact_sha(file) == shard["sha256"]
    handoff_files["vectors/" + shard["path"]] = file
for shard in handoff_manifest["train_shards"] + [handoff_manifest["candidates"]] + handoff_manifest.get("extra_files", []):
    file = handoff_prepared / shard["path"]
    assert artifact_sha(file) == shard["sha256"]
    handoff_files["prepared/" + shard["path"]] = file
metrics = handoff_run / "metrics.jsonl"
if metrics.is_file():
    handoff_files["model/metrics.jsonl"] = metrics
handoff_inventory = {name: {"sha256": artifact_sha(file), "bytes": file.stat().st_size}
                     for name, file in handoff_files.items()}
handoff_zip = Path("/content/SAR-Pure-model-vectors-b2ed5728b9a2.zip")
handoff_readme = """# Pure model and vectors handoff
Run: https://wandb.ai/SAR-Lab/kuairand-two-tower/runs/b2ed5728b9a2
Completed 1 epoch / 124780 training positives; validation in-batch loss 5.516606385154798.
model/: actual checkpoint, completion marker, run manifest and config.
vectors/: 7583 normalized 64-dimensional item vectors in two NPZ shards, with ID arrays and manifest.
prepared/: exact manifest and referenced tables from this training runtime. Preserve these bytes for checkpoint binding.
source/: the immutable source ZIP and checksum used in this run. Use its uv.lock environment.
Tests are separately published as datasets/SAR-Pure-test-handoff-v1.zip.
No training was performed by this copy step. Originals remain in personal Drive.
For inference build_model(config, prepared manifest), load trusted checkpoint with torch.load(..., map_location='cpu', weights_only=False), then model.load_state_dict(state['model']); model.eval(); use torch.no_grad().
Use checkpoint's shared Item Tower to encode each history; left-pad raw IDs with -1 and set mask False. Unknown users map to index zero.
Scores are normalized user-vector dot product with exported item vectors; align using item IDs.
Original absolute paths in run_manifest are provenance; use model/<checkpoint filename> after unpacking.
The prepared train/validation tables are included for manifest checksum compatibility; do NOT include test in optimization.
Test protocol allows strictly earlier observed test events in later request histories, not parameter updates.
Recommendation quality still needs Recall@100/NDCG@100 vs a train-popularity baseline.
"""
with zipfile.ZipFile(handoff_zip, "w", zipfile.ZIP_DEFLATED) as archive:
    for name, file in sorted(handoff_files.items()):
        archive.write(file, name)
    archive.writestr("README.md", handoff_readme)
    archive.writestr("checksums.json", json.dumps(handoff_inventory, indent=2))
with zipfile.ZipFile(handoff_zip) as archive:
    assert archive.testzip() is None
handoff_credentials, _ = google.auth.default()
handoff_service = build("drive", "v3", credentials=handoff_credentials, cache_discovery=False)
handoff_parent = "18756Hf7J_8WW8e_hURw3I2qd4pDmP8ZG"
handoff_folder = handoff_service.files().get(fileId=handoff_parent, fields="id,name,mimeType,shared,shortcutDetails", supportsAllDrives=True).execute()
print("Verified UI destination metadata:", json.dumps(handoff_folder))
assert handoff_folder["name"].strip() == "releases", "Destination name differs"
for _ in range(5):
    if handoff_folder["mimeType"] != "application/vnd.google-apps.shortcut":
        break
    handoff_parent = handoff_folder["shortcutDetails"]["targetId"]
    handoff_folder = handoff_service.files().get(fileId=handoff_parent, fields="id,name,mimeType,shared,shortcutDetails", supportsAllDrives=True).execute()
assert handoff_folder["mimeType"] == "application/vnd.google-apps.folder", "Destination is not a folder"
print("Resolved shared folder:", json.dumps(handoff_folder))
handoff_existing = handoff_service.files().list(
    q=f"'{handoff_parent}' in parents and name='{handoff_zip.name}' and trashed=false",
    fields="files(id,name,md5Checksum)", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()["files"]
handoff_md5 = hashlib.md5(handoff_zip.read_bytes()).hexdigest()
if handoff_existing:
    assert len(handoff_existing) == 1 and handoff_existing[0]["md5Checksum"] == handoff_md5, "Existing package differs; do not overwrite"
    handoff_uploaded = handoff_existing[0]
else:
    handoff_uploaded = handoff_service.files().create(
        body={"name": handoff_zip.name, "parents": [handoff_parent]},
        media_body=MediaFileUpload(str(handoff_zip), mimetype="application/zip", resumable=True),
        fields="id,name,size,md5Checksum,webViewLink", supportsAllDrives=True).execute()
handoff_verified = handoff_service.files().get(
    fileId=handoff_uploaded["id"], fields="id,name,size,md5Checksum,parents,webViewLink,shared",
    supportsAllDrives=True).execute()
assert handoff_verified["md5Checksum"] == handoff_md5
assert int(handoff_verified["size"]) == handoff_zip.stat().st_size
assert handoff_parent in handoff_verified["parents"]
print(json.dumps({"status": "copied_and_verified", "file": handoff_verified,
                  "package_sha256": artifact_sha(handoff_zip),
                  "release_folder": handoff_parent, "originals_preserved": True,
                  "training_started_by_copy": False}, indent=2))
