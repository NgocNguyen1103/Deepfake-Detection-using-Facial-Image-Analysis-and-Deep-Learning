from DataLoader import build_dataloaders


def main():
    root_dir = r"D:\M1\M1 Internship\preprocessed_dataset"

    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        image_size=224,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    print("Batch keys:", batch.keys())
    print("Image shape:", batch["image"].shape)
    print("Label shape:", batch["label"].shape)
    print("Labels:", batch["label"])

    if "video_id" in batch:
        print("Video IDs:", batch["video_id"][:3])

    if "frame_num" in batch:
        print("Frame nums:", batch["frame_num"][:3])


if __name__ == "__main__":
    main()