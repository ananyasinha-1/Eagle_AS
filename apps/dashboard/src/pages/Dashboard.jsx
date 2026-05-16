import CameraCard from "../components/CameraCard"

export default function Dashboard() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 h-screen bg-black">

      <CameraCard title="Camera 1" trackId="P-101" />

      <CameraCard title="Camera 2" trackId="P-102" />

      <CameraCard title="Camera 3" trackId="P-101" />

      <CameraCard title="Camera 4" trackId="P-103" />

    </div>
  )
}