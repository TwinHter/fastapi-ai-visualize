// App.tsx
import { Box, LinearProgress } from "@mui/material";
import Sidebar from "./sidebar";
import { Outlet } from "react-router";
import { Observer } from "mobx-react-lite";
import { useStore } from "../../lib/hook/useStore";
function App() {
    const { uiStore } = useStore();
    return (
        <>
            <Observer>
                {() => (uiStore.isLoading ? <LinearProgress /> : null)}
            </Observer>
            <Box display="flex">
                <Box width="25%">
                    <Sidebar />
                </Box>
                <Box width="75%">
                    <Outlet />
                </Box>
            </Box>
        </>
    );
}

export default App;
