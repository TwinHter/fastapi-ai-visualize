import {
    Box,
    Typography,
    Card,
    CardContent,
    CardHeader,
    Grid,
    List,
    ListItem,
    ListItemText,
    Divider,
} from "@mui/material";
import PeopleIcon from "@mui/icons-material/People";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import MemoryIcon from "@mui/icons-material/Memory";
import { HomePageContent } from "../lib/types/const-index.t";

export default function HomePage() {
    return (
        <Box
            sx={{ p: { xs: 2, sm: 4 }, bgcolor: "#f0f4f8", minHeight: "100vh" }}
        >
            {/* Title & Description */}
            <Box sx={{ mb: 4 }}>
                <Typography
                    variant="h3"
                    component="h1"
                    gutterBottom
                    sx={{ fontWeight: "bold", color: "#1976d2" }}
                >
                    {HomePageContent.title}
                </Typography>
                <Typography
                    variant="subtitle1"
                    color="text.secondary"
                    sx={{ maxWidth: 800 }}
                >
                    {HomePageContent.description}
                </Typography>
            </Box>

            {/* About Project */}
            <Card sx={{ mb: 4, borderRadius: 3, boxShadow: 3 }}>
                <CardHeader
                    avatar={<MenuBookIcon color="primary" />}
                    title="About the Project"
                    titleTypographyProps={{ variant: "h6", fontWeight: "bold" }}
                />
                <CardContent>
                    <Typography variant="subtitle1" color="text.secondary">
                        {HomePageContent.aboutProject}
                    </Typography>
                </CardContent>
            </Card>

            {/* Team Members */}
            <Card sx={{ mb: 4, borderRadius: 3, boxShadow: 3 }}>
                <CardHeader
                    avatar={<PeopleIcon color="primary" />}
                    title="Meet the Team"
                    subheader="Group 1"
                    titleTypographyProps={{ variant: "h6", fontWeight: "bold" }}
                />
                <CardContent>
                    <Grid container spacing={3}>
                        {HomePageContent.teamMembers.map((member) => (
                            <Box display="flex" flexWrap="wrap" key={member.id}>
                                <Box
                                    textAlign="center"
                                    sx={{
                                        p: 2,
                                        bgcolor: "#ffffff",
                                        borderRadius: 2,
                                        boxShadow: 1,
                                    }}
                                >
                                    <Typography
                                        variant="subtitle1"
                                        fontWeight="bold"
                                    >
                                        {member.name}
                                    </Typography>
                                    <Typography
                                        variant="body2"
                                        color="text.secondary"
                                    >
                                        {member.id}
                                    </Typography>
                                </Box>
                            </Box>
                        ))}
                    </Grid>
                </CardContent>
            </Card>

            {/* Core Features */}
            <Card sx={{ borderRadius: 3, boxShadow: 3 }}>
                <CardHeader
                    avatar={<MemoryIcon color="primary" />}
                    title="Core Features"
                    titleTypographyProps={{ variant: "h6", fontWeight: "bold" }}
                />
                <CardContent>
                    <List>
                        {HomePageContent.coreFeature.map((feature, index) => (
                            <Box
                                key={feature.title}
                                sx={{
                                    mb: 3,
                                    p: 2,
                                    bgcolor: "#f9f9f9",
                                    borderRadius: 2,
                                    transition: "transform 0.2s",
                                    "&:hover": {
                                        transform: "scale(1.01)",
                                        boxShadow: 2,
                                    },
                                }}
                            >
                                <Typography variant="h6" gutterBottom>
                                    {feature.title}
                                </Typography>
                                <List dense disablePadding>
                                    {feature.items.map((desc, idx) => (
                                        <ListItem key={idx} sx={{ py: 0.5 }}>
                                            <ListItemText
                                                primary={
                                                    <Typography
                                                        variant="body2"
                                                        color="text.secondary"
                                                    >
                                                        <b>{desc.name}:</b>{" "}
                                                        {desc.desc}
                                                    </Typography>
                                                }
                                            />
                                        </ListItem>
                                    ))}
                                </List>
                                {index <
                                    HomePageContent.coreFeature.length - 1 && (
                                    <Divider
                                        sx={{ mt: 2, borderColor: "#e0e0e0" }}
                                    />
                                )}
                            </Box>
                        ))}
                    </List>
                </CardContent>
            </Card>
        </Box>
    );
}
