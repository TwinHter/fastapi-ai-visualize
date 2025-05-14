// src/features/AlgorithmPage.tsx

import {
    Box,
    Typography,
    FormControl,
    Divider,
    Button,
    Accordion,
    AccordionDetails,
    AccordionSummary,
} from "@mui/material";
import { useState } from "react";
import { useMetric } from "../../lib/hook/useMetric";
import { MetricsTable } from "../../app/shared/MetricsTable";
import SelectInput from "../../app/shared/SelectInput";
import { algorithmCodes, algorithms } from "../../lib/types/const-index.t";
import { useNavigate } from "react-router";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { extractAlgo } from "../../lib/utils/extractAlgo";
import { usePredict } from "../../lib/hook/usePredict";

export default function Algorithm() {
    const navigate = useNavigate();
    const [selectedAlgo, setSelectedAlgo] = useState("tree_custom");

    const { name, type } = extractAlgo(selectedAlgo);

    const { metric: algorithmMetrics } = useMetric(name, type);
    const { trainAll } = usePredict();
    const handleTrainAll = async () => {
        await trainAll.mutateAsync();
    };
    return (
        <Box sx={{ padding: 4 }}>
            <Typography variant="h4" gutterBottom>
                About Algorithm
            </Typography>

            <Box
                sx={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    flexWrap: "wrap",
                    gap: 2,
                    my: 3,
                }}
            >
                <FormControl
                    sx={{
                        minWidth: 240,
                        flex: 1,
                    }}
                >
                    <SelectInput
                        label="Choose Algorithm"
                        value={selectedAlgo}
                        onChange={(e) => setSelectedAlgo(e.target.value)}
                        options={[...algorithms]}
                        id="algo-select"
                    />
                </FormControl>

                <Button
                    variant="contained"
                    onClick={() => navigate(`/train/${name}/${type}`)}
                    sx={{
                        whiteSpace: "nowrap",
                        height: "56px",
                        bgcolor: "primary.main",
                        ":hover": {
                            bgcolor: "primary.dark",
                        },
                    }}
                >
                    Train Algorithm
                </Button>
                <Button
                    variant="contained"
                    onClick={() => handleTrainAll()}
                    sx={{
                        whiteSpace: "nowrap",
                        height: "56px",
                        bgcolor: "primary.main",
                        ":hover": {
                            bgcolor: "primary.dark",
                        },
                    }}
                >
                    Train All
                </Button>
            </Box>

            <Divider sx={{ my: 3 }} />

            <Accordion
                key="Pseudocode"
                sx={{
                    mb: 2,
                    border: "1px solid #ddd",
                    borderRadius: 1,
                    boxShadow: "0 2px 6px rgba(0,0,0,0.08)",
                    "&::before": {
                        display: "none",
                    },
                }}
            >
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    sx={{
                        bgcolor: "#00e5ff",
                        fontWeight: 600,
                    }}
                >
                    <Typography
                        sx={{
                            fontSize: "1.1rem",
                            fontWeight: 600,
                            color: "#333",
                        }}
                    >
                        Pseudocode
                    </Typography>
                </AccordionSummary>
                <AccordionDetails>
                    <SyntaxHighlighter language="python" style={materialDark}>
                        {algorithmCodes[selectedAlgo]}
                    </SyntaxHighlighter>
                </AccordionDetails>
            </Accordion>

            <Divider sx={{ my: 3 }} />

            <Typography variant="h5" gutterBottom>
                Result
            </Typography>

            <MetricsTable algorithmMetrics={algorithmMetrics} />
        </Box>
    );
}
