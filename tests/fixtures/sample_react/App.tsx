import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { UserCard } from "./UserCard";
import axios from "axios";

const router = createBrowserRouter([
  {
    path: "/",
    element: <div>Home</div>,
  },
  {
    path: "/users/:userId",
    element: <UserCard userId="1" />,
  },
  {
    path: "/users",
    element: <div>User List</div>,
  },
]);

async function fetchUsers() {
  const response = await axios.get("/api/v1/users/");
  return response.data;
}

async function createUser(data: object) {
  const response = await axios.post("/api/v1/users/", data);
  return response.data;
}

export default function App() {
  return <RouterProvider router={router} />;
}
