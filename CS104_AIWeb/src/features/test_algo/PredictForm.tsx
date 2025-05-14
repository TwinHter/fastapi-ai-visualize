// src/features/AlgorithmPage.tsx

import {
    Box,
    Typography,
    FormControl,
    Divider,
    Button,
    Alert,
} from "@mui/material";
import { useEffect, useState } from "react";
import { testSchema, type TestSchema } from "../../lib/schema/testSchema";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import SelectInput from "../../app/shared/SelectInput";
import { algorithms, countryList, itemList } from "../../lib/types/const-index.t";
import type { Feature, PredictValue } from "../../lib/types/index.t";
import { usePredict } from "../../lib/hook/usePredict";
import TextInput from "../../app/shared/TextInput";
import SelectFormInput from "../../app/shared/SelectFormInput";

export default function PredictForm() {
    const { handleSubmit, control } = useForm<TestSchema>({
        resolver: zodResolver(testSchema),
        mode: "onTouched",
    });
    const [selectedAlgo, setSelectedAlgo] = useState("tree_custom");

    const [selectedType, setSelectedType] = useState("custom");
    const [selectedName, setSelectedName] = useState("decision_tree");

    useEffect(() => {
        switch (selectedAlgo) {
            case "tree_custom":
                setSelectedType("custom");
                setSelectedName("decision_tree");
                break;
            case "tree_sklearn":
                setSelectedType("sklearn");
                setSelectedName("decision_tree");
                break;
            case "lasso_custom":
                setSelectedType("custom");
                setSelectedName("lasso");
                break;
            case "lasso_sklearn":
                setSelectedType("sklearn");
                setSelectedName("lasso");
                break;
            default:
                break;
        }
    }, [selectedAlgo]);

    const { predictData } = usePredict(selectedName, selectedType);
    const [prediction, setPrediction] = useState<PredictValue | null>(null);

    const onSubmit = async (data: TestSchema) => {
        const feature: Feature = {
            Area: data.Area,
            Item: data.Item,
            Year: data.Year,
            average_rain_fall_mm_per_year: data.average_rain_fall_mm_per_year,
            pesticides_tonnes: data.pesticides_tonnes,
            avg_temp: data.avg_temp,
        };
        try {
            predictData.mutateAsync(feature).then((res) => {
                if (res) {
                    setPrediction(res);
                } else {
                    setPrediction(null);
                }
            });
        } catch (error) {
            console.error("Error during prediction:", error);
            setPrediction(null);
        }
    };

    return (
        <Box sx={{ padding: 4 }}>
            <Typography variant="h4" gutterBottom>
                Predict Algorithm
            </Typography>

            <FormControl sx={{ minWidth: 240, my: 2 }}>
                <SelectInput
                    options={[...algorithms]}
                    value={selectedAlgo}
                    onChange={(e) => setSelectedAlgo(e.target.value)}
                    label="Choose Test Algorithm"
                />
            </FormControl>

            <Divider sx={{ my: 3 }} />

            <Typography variant="h5" gutterBottom>
                Predict
            </Typography>

            <Box
                component="form"
                onSubmit={handleSubmit(onSubmit)}
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                    maxWidth: 400,
                }}
            >
                <SelectFormInput
                    label="Area"
                    name="Area"
                    control={control}
                    items={countryList.map((country) => ({
                        text: country,
                        value: country,
                    }))}
                />
                <SelectFormInput
                    label="Item"
                    name="Item"
                    control={control}
                    items={itemList.map((item) => ({
                        text: item,
                        value: item,
                    }))}
                />

                <TextInput name="Year" control={control} label="Year" />
                <TextInput
                    name="average_rain_fall_mm_per_year"
                    control={control}
                    label="Average Rainfall (mm)"
                />
                <TextInput
                    name="pesticides_tonnes"
                    control={control}
                    label="Pesticides (tonnes)"
                />
                <TextInput
                    name="avg_temp"
                    control={control}
                    label="Average Temperature"
                />

                <Button type="submit" variant="contained" color="primary">
                    Predict
                </Button>

                {prediction !== null ? (
                    <Alert severity="success">
                        Result: <strong>{prediction.hg_ha_yield}</strong>
                    </Alert>
                ) : (
                    <Alert severity="info">
                        Result: <strong>Waiting for prediction...</strong>
                    </Alert>
                )}
            </Box>
        </Box>
    );
}
