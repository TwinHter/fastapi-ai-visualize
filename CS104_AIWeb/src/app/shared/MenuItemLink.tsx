import { MenuItem } from "@mui/material";
import { NavLink } from "react-router";

export default function MenuItemLink({
    children,
    to,
}: {
    children: React.ReactNode;
    to: string;
}) {
    return (
        <MenuItem
            component={NavLink}
            to={to}
            sx={{
                color: "inherit",
                textDecoration: "none",
                "&.active": { bgcolor: "primary.main" },
                fontSize: "1.2rem",
                textTransform: "uppercase",
                fontWeight: 500,
            }}
        >
            {children}
        </MenuItem>
    );
}
