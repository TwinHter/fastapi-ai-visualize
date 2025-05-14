import { useState } from "react";
import {
    Box,
    Typography,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Button,
    Card,
    CardContent,
    CardHeader,
    CardMedia,
} from "@mui/material";
import axios from "axios";
import { countryList, itemList } from "../lib/types/const-index.t";

// your lists; import them or define here

export default function EdaVisualize() {
    // state for Area+Item form
    const [aiArea, setAiArea] = useState<string>("");
    const [aiItem, setAiItem] = useState<string>("");
    const [aiImgUrl, setAiImgUrl] = useState<string>("");

    // state for Area‐only form
    const [aArea, setAArea] = useState<string>("");
    const [aImgUrl, setAImgUrl] = useState<string>("");

    // fetch blob & convert to object URL
    const fetchImage = async (url: string, params: Record<string, string>) => {
        try {
            const res = await axios.get(url, {
                params,
                responseType: "blob",
            });
            return URL.createObjectURL(res.data);
        } catch (err) {
            console.error("Failed to load image", err);
            return "";
        }
    };

    const handleAiSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!aiArea || !aiItem) return;
        const url = await fetchImage("http://localhost:8000/eda/area_item", {
            area: aiArea,
            item: aiItem,
        });
        setAiImgUrl(url);
    };

    const handleASubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!aArea) return;
        const url = await fetchImage("http://localhost:8000/eda/area", {
            area: aArea,
        });
        setAImgUrl(url);
    };

    return (
        <Box sx={{ p: 4, bgcolor: "#f9f9f9" }}>
            <Typography variant="h4" gutterBottom>
                EDA Visualize
            </Typography>

            {/* Area + Item */}
            <Card sx={{ mb: 4, boxShadow: 2 }}>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        By Area & Item
                    </Typography>
                    <Box
                        component="form"
                        onSubmit={handleAiSubmit}
                        sx={{
                            display: "flex",
                            gap: 2,
                            alignItems: "center",
                            flexWrap: "wrap",
                        }}
                    >
                        <FormControl sx={{ minWidth: 160 }}>
                            <InputLabel>Area</InputLabel>
                            <Select
                                value={aiArea}
                                label="Area"
                                onChange={(e) =>
                                    setAiArea(e.target.value as string)
                                }
                            >
                                {countryList.map((c) => (
                                    <MenuItem key={c} value={c}>
                                        {c}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <FormControl sx={{ minWidth: 160 }}>
                            <InputLabel>Item</InputLabel>
                            <Select
                                value={aiItem}
                                label="Item"
                                onChange={(e) =>
                                    setAiItem(e.target.value as string)
                                }
                            >
                                {itemList.map((it) => (
                                    <MenuItem key={it} value={it}>
                                        {it}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <Button type="submit" variant="contained">
                            Submit
                        </Button>
                    </Box>

                    {aiImgUrl && (
                        <Box sx={{ mt: 3 }}>
                            <img
                                src={aiImgUrl}
                                alt="EDA Area+Item"
                                style={{
                                    maxWidth: "100%",
                                    borderRadius: 4,
                                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                                }}
                            />
                        </Box>
                    )}
                </CardContent>
            </Card>

            {/* Area only */}
            <Card>
                <CardContent>
                    <Typography variant="h6" gutterBottom>
                        By Area Only
                    </Typography>
                    <Box
                        component="form"
                        onSubmit={handleASubmit}
                        sx={{
                            display: "flex",
                            gap: 2,
                            alignItems: "center",
                            flexWrap: "wrap",
                        }}
                    >
                        <FormControl sx={{ minWidth: 160 }}>
                            <InputLabel>Area</InputLabel>
                            <Select
                                value={aArea}
                                label="Area"
                                onChange={(e) =>
                                    setAArea(e.target.value as string)
                                }
                            >
                                {countryList.map((c) => (
                                    <MenuItem key={c} value={c}>
                                        {c}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <Button type="submit" variant="contained">
                            Submit
                        </Button>
                    </Box>

                    {aImgUrl && (
                        <Box sx={{ mt: 3 }}>
                            <img
                                src={aImgUrl}
                                alt="EDA Area"
                                style={{
                                    maxWidth: "100%",
                                    borderRadius: 4,
                                    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                                }}
                            />
                        </Box>
                    )}
                </CardContent>
            </Card>

            <Card sx={{ mb: 4, borderRadius: 3, boxShadow: 3 }}>
                <CardHeader
                    title="Heat Map"
                    titleTypographyProps={{ variant: "h6", fontWeight: "bold" }}
                />
                {/* Cách 1: Dùng CardMedia */}
                <CardMedia
                    component="img"
                    image="./public/heat_map.png"
                    alt="EDA Heat Map"
                    sx={{
                        maxHeight: 600,
                        objectFit: "contain",
                    }}
                />
            </Card>
        </Box>
    );
}
