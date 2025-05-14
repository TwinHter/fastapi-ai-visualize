import axios from "axios";
import { useState, useEffect } from "react";

function PlotImage() {
    const [imageSrc, setImageSrc] = useState("");

    useEffect(() => {
        axios
            .get("http://localhost:8000/plot-image", {
                responseType: "blob",
            })
            .then((res) => {
                const imageURL = URL.createObjectURL(res.data); // 👈 tạo URL cho blob
                setImageSrc(imageURL);
            })
            .catch((error) => {
                console.error("Failed to load image:", error);
            });
    }, []);

    return (
        <div>
            <h2>Biểu đồ</h2>
            {imageSrc ? (
                <img src={imageSrc} alt="plot" style={{ maxWidth: "100%" }} />
            ) : (
                <p>Đang tải ảnh...</p>
            )}
        </div>
    );
}

export default PlotImage;
