import { Box, Typography, Button, FormControl, Grid } from "@mui/material";
import { useState } from "react";
import SelectInput from "../../app/shared/SelectInput";
import axios from "axios";

export default function CompareAlgorithm() {
    const [metric, setMetric] = useState("mse");
    const [pltType, setPltType] = useState("line");
    const [imageSrc, setImageSrc] = useState("");

    const handleCompare = () => {
        axios
            .get("http://localhost:8000/compare", {
                params: {
                    metric: metric,
                    type: pltType,
                },
                responseType: "blob",
            })
            .then((res) => {
                const imageURL = URL.createObjectURL(res.data);
                setImageSrc(imageURL);
            })
            .catch((error) => {
                console.error("Failed to load image:", error);
            });
    };
    return (
        <Box sx={{ padding: 4 }}>
            <Typography variant="h4" gutterBottom>
                Compare Algorithms
            </Typography>

            <Grid container spacing={2} alignItems="center">
                <Grid size={4}>
                    <FormControl fullWidth>
                        <SelectInput
                            label="Algorithm 1"
                            value={metric}
                            onChange={(e) => setMetric(e.target.value)}
                            options={[
                                { label: "MSE", value: "mse" },
                                { label: "R2", value: "r2" },
                            ]}
                        />
                    </FormControl>
                </Grid>

                <Grid size={4}>
                    <FormControl fullWidth>
                        <SelectInput
                            label="Algorithm 2"
                            value={pltType}
                            onChange={(e) => setPltType(e.target.value)}
                            options={[
                                { label: "Bar Plot", value: "bar" },
                                { label: "Line Plot", value: "line" },
                                { label: "Scatter Plot", value: "scatter" },
                                { label: "Radar Plot", value: "radar" },
                            ]}
                        />
                    </FormControl>
                </Grid>
            </Grid>

            <Button variant="contained" onClick={handleCompare} sx={{ mt: 3 }}>
                Compare
            </Button>

            {imageSrc ? (
                <img src={imageSrc} alt="plot" style={{ maxWidth: "100%" }} />
            ) : (
                <p>Loading...</p>
            )}
        </Box>
    );
}
