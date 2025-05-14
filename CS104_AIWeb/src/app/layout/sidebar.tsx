import { Drawer, Toolbar, Divider, Typography, Box } from "@mui/material";
import HomeIcon from "@mui/icons-material/Home";
import MenuItemLink from "../shared/MenuItemLink";
import DatasetIcon from "@mui/icons-material/Storage";
import AlgorithmIcon from "@mui/icons-material/Functions";
import TestIcon from "@mui/icons-material/PlayArrow";
import CompareIcon from "@mui/icons-material/Compare";

const drawerWidth = 350;

const navItems = [
    { text: "Home Page", icon: <HomeIcon />, path: "/" },
    { text: "Dataset", icon: <DatasetIcon />, path: "/data-intro" },
    { text: "Algorithm", icon: <AlgorithmIcon />, path: "/algorithm" },
    { text: "Test", icon: <TestIcon />, path: "/predict" },
    { text: "Compare Algorithm", icon: <CompareIcon />, path: "/compare-algo" },
];

export default function Sidebar() {
    return (
        <Drawer
            variant="permanent"
            sx={{
                width: drawerWidth,
                flexShrink: 0,
                [`& .MuiDrawer-paper`]: {
                    width: drawerWidth,
                    boxSizing: "border-box",
                },
            }}
        >
            <Toolbar>
                <Typography variant="h6">Side Bar</Typography>
            </Toolbar>
            <Divider />
            <Box sx={{ overflow: "auto" }}>
                {navItems.map((item) => (
                    <MenuItemLink key={item.path} to={item.path}>
                        {item.icon} {item.text}
                    </MenuItemLink>
                ))}
            </Box>
        </Drawer>
    );
}
