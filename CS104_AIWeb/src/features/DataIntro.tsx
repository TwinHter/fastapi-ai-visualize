import {
    Box,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Stack,
    Paper,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { dataIntroduction } from "../lib/types/const-index.t";
import EdaVisualize from "./EdaVisualize";

export default function DataIntro() {
    return (
        <Box
            sx={{
                p: { xs: 2, md: 4 },
                bgcolor: "#f8f9fa",
                fontFamily: "'Roboto Slab', serif",
                minHeight: "100vh",
            }}
        >
            <Typography
                variant="h3"
                component="h1"
                gutterBottom
                sx={{
                    fontWeight: 700,
                    color: "#2c3e50",
                    textAlign: "center",
                    mb: 4,
                }}
            >
                {dataIntroduction.name}
            </Typography>

            {/** Accordion Block Component */}
            {[
                {
                    title: "Introduction",
                    content: (
                        <Typography
                            variant="body1"
                            sx={{ fontSize: "1rem", color: "#555" }}
                        >
                            {dataIntroduction.description}
                        </Typography>
                    ),
                },
                {
                    title: "Features",
                    content: (
                        <TableContainer component={Paper} elevation={2}>
                            <Table>
                                <TableHead>
                                    <TableRow sx={{ bgcolor: "#f0f4f8" }}>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Type</TableCell>
                                        <TableCell>Description</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {dataIntroduction.features.map((f) => (
                                        <TableRow key={f.name}>
                                            <TableCell>{f.name}</TableCell>
                                            <TableCell>{f.type}</TableCell>
                                            <TableCell>
                                                {f.description}
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    ),
                },
                {
                    title: "Target",
                    content: (
                        <TableContainer component={Paper} elevation={2}>
                            <Table>
                                <TableHead>
                                    <TableRow sx={{ bgcolor: "#f0f4f8" }}>
                                        <TableCell>Name</TableCell>
                                        <TableCell>Type</TableCell>
                                        <TableCell>Description</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    <TableRow
                                        key={dataIntroduction.target.name}
                                    >
                                        <TableCell>
                                            {dataIntroduction.target.name}
                                        </TableCell>
                                        <TableCell>
                                            {dataIntroduction.target.type}
                                        </TableCell>
                                        <TableCell>
                                            {
                                                dataIntroduction.target
                                                    .description
                                            }
                                        </TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>
                        </TableContainer>
                    ),
                },
                {
                    title: "Statistics",
                    content: (
                        <Stack spacing={2} sx={{ mt: 1 }}>
                            {dataIntroduction.statistics.map((s) => (
                                <Stack
                                    direction="row"
                                    spacing={2}
                                    key={s.name}
                                    alignItems="center"
                                >
                                    <Typography sx={{ minWidth: 120 }}>
                                        {s.name}:
                                    </Typography>
                                    <Chip label={s.value} color="primary" />
                                </Stack>
                            ))}
                        </Stack>
                    ),
                },
                {
                    title: "Metrics",
                    content: (
                        <Stack direction="row" spacing={1} flexWrap="wrap">
                            {dataIntroduction.metrics.map((m) => (
                                <Chip
                                    label={m.name}
                                    key={m.name}
                                    color="secondary"
                                    variant="outlined"
                                />
                            ))}
                        </Stack>
                    ),
                },
                {
                    title: "Sample",
                    content: (
                        <TableContainer component={Paper} elevation={2}>
                            <Table size="small">
                                <TableHead>
                                    <TableRow sx={{ bgcolor: "#f0f4f8" }}>
                                        {Object.keys(
                                            dataIntroduction.sample[0]
                                        ).map((key) => (
                                            <TableCell key={key}>
                                                {key}
                                            </TableCell>
                                        ))}
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {dataIntroduction.sample.map((s, index) => (
                                        <TableRow key={index}>
                                            {Object.values(s).map(
                                                (value, i) => (
                                                    <TableCell key={i}>
                                                        {value}
                                                    </TableCell>
                                                )
                                            )}
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    ),
                },
                {
                    title: "EDA",
                    content: <EdaVisualize />,
                },
                {
                    title: "Preprocessing",
                    content: (
                        <>
                            <Typography variant="body1" paragraph>
                                There are two categorical columns in the
                                dataset: <strong>items</strong> and{" "}
                                <strong>countries</strong>. Categorical
                                variables contain label values (e.g., names or
                                codes) rather than numeric values, and their
                                possible values are typically limited to a fixed
                                set. In this case, the values represent specific
                                items and countries.
                            </Typography>
                            <Typography variant="body1" paragraph>
                                <strong>Mean Target Encoding</strong> is a
                                technique used to convert these categorical
                                variables into numerical values by replacing
                                each category with the mean of the target
                                variable corresponding to that category.
                            </Typography>
                            <Typography variant="body1" paragraph>
                                <strong>Normalization/Standardization</strong>{" "}
                                is a separate preprocessing step applied to the
                                (now fully numeric) feature set to ensure that
                                different features are on comparable scales—this
                                is important for some models but not others:
                            </Typography>
                            <Typography
                                variant="body2"
                                component="div"
                                sx={{ ml: 2 }}
                            >
                                <ul>
                                    <li>
                                        <strong>Decision Tree:</strong> No
                                        scaling required, since tree-based
                                        models are scale-invariant (they split
                                        on thresholds, not distances or
                                        gradients).
                                    </li>
                                    <li>
                                        <strong>Lasso Regression:</strong> Only
                                        the input features <code>X</code> are
                                        standardized (zero mean, unit variance).
                                        This is necessary because Lasso’s L1
                                        penalty is sensitive to feature scale.
                                    </li>
                                    <li>
                                        <strong>
                                            Artificial Neural Network (ANN):
                                        </strong>{" "}
                                        Both <code>X</code> and <code>y</code>{" "}
                                        are standardized, because neural
                                        networks use gradient-based optimization
                                        and converge faster and more stably when
                                        features (and targets, for regression)
                                        share a common scale.
                                    </li>
                                </ul>
                            </Typography>
                        </>
                    ),
                },
            ].map((section, index) => (
                <Accordion
                    key={section.title}
                    defaultExpanded={index === 0}
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
                            {section.title}
                        </Typography>
                    </AccordionSummary>
                    <AccordionDetails>{section.content}</AccordionDetails>
                </Accordion>
            ))}
        </Box>
    );
}
