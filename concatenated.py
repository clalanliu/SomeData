

[abstract_factory.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import (
    Bluetooth,
    WiFi,
    SmartDevice,
    CommunicationTechnology,
)

logger = logging.getLogger(__name__)


class SmartLight(SmartDevice):
    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

    def operate(self):
        super().operate()
        self.turn_on()
        self.turn_off()


class SmartThermostat(SmartDevice):
    @abstractmethod
    def increase_temperature(self):
        pass

    @abstractmethod
    def decrease_temperature(self):
        pass

    def __init__(self, communication: CommunicationTechnology):
        super().__init__(communication)
        self.control_strategy = None

    def operate(self):
        super().operate()
        self.increase_temperature()
        self.decrease_temperature()

    def set_temperature_control_strategy(self, control_strategy):
        self.control_strategy = control_strategy(
            increase_func=self.increase_temperature,
            decrease_func=self.decrease_temperature,
        )

    def control_temperature(self):
        if self.control_strategy:
            self.control_strategy.control_temperature()


class LivingRoomSmartLight(SmartLight):
    def turn_on(self):
        logger.info("Living Room's Smart Light is turned on.")

    def turn_off(self):
        logger.info("Living Room's Smart Light is turned off.")


class KitchenRoomSmartLight(SmartLight):
    def turn_on(self):
        logger.info("Kitchen Room's Smart Light is turned on.")

    def turn_off(self):
        logger.info("Kitchen Room's Smart Light is turned off.")


class LivingRoomSmartThermostat(SmartThermostat):
    def increase_temperature(self):
        logger.info("Living Room's Smart Thermostat is increasing temperature.")

    def decrease_temperature(self):
        logger.info("Living Room's Smart Thermostat is decreasing temperature.")

    def set_temperature(self, temp):
        logger.info(f"Living Room's Smart Thermostat is set to {temp}")


class KitchenRoomSmartThermostat(SmartThermostat):
    def increase_temperature(self):
        logger.info("Kitchen Room's Smart Thermostat is increasing temperature.")

    def decrease_temperature(self):
        logger.info("Kitchen Room's Smart Thermostat is decreasing temperature.")


class SmartDeviceFactory(ABC):
    @abstractmethod
    def create_smart_light(self):
        pass

    @abstractmethod
    def create_smart_thermostat(self):
        pass


class LivingRoomDeviceFactory(SmartDeviceFactory):
    def create_smart_light(self):
        return LivingRoomSmartLight(WiFi())

    def create_smart_thermostat(self):
        return LivingRoomSmartThermostat(Bluetooth())


class KitchenRoomDeviceFactory(SmartDeviceFactory):
    def create_smart_light(self):
        return KitchenRoomSmartLight(WiFi())

    def create_smart_thermostat(self):
        return KitchenRoomSmartThermostat(Bluetooth())


[adapter.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartDevice, WiFi

logger = logging.getLogger(__name__)


class ThirdPartySmartDevice(ABC):
    """Third party devices don't have the same interface as our SmartDevice."""

    @abstractmethod
    def switch_on(self):
        pass

    @abstractmethod
    def switch_off(self):
        pass


class ThirdPartySmartLight(ThirdPartySmartDevice):
    def switch_on(self):
        logger.info("ThirdPartySmartLight is turned on.")

    def switch_off(self):
        logger.info("ThirdPartySmartLight is turned off.")


class SmartDeviceAdapter(SmartDevice):
    """This adapter wraps a third party device so it can be used in our system."""

    def __init__(self, third_party_device):
        super().__init__(communication=WiFi())
        self.third_party_device = third_party_device

    def operate(self):
        super().operate()
        self.third_party_device.switch_on()
        self.third_party_device.switch_off()


[bridge.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartDevicePrototype

logger = logging.getLogger(__name__)


class CommunicationTechnology(ABC):
    """Implementor interface for communication technology."""

    @abstractmethod
    def communicate(self):
        pass


class Bluetooth(CommunicationTechnology):
    def communicate(self):
        logger.info("Communicating via Bluetooth.")


class WiFi(CommunicationTechnology):
    def communicate(self):
        logger.info("Communicating via Wi-Fi.")


# Base SmartDevice class
# Update SmartDevice and its children
class SmartDevice(ABC, SmartDevicePrototype):
    def __init__(self, communication):
        self.observers = []
        self.communication = communication

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify(self, event):
        for observer in self.observers:
            observer.update(self, event)

    def operate(self):
        self.notify(
            f"event from device {id(self)}"
        )  # Notify observers when device is operated
        self.communication.communicate()

    def accept(self, visitor):
        visitor.visit(self)


[builder.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartDeviceFactory
from SmartHomeSystem import PaintingFactory
from SmartHomeSystem import DeviceIterator
from SmartHomeSystem import RoomState

logger = logging.getLogger(__name__)
painting_factory = PaintingFactory()


class Room:
    def __init__(self, name):
        self.name = name
        self.walls = None
        self.doors = None
        self.windows = None
        self.devices = []
        self.paintings = []
        self._mementos = []

    def add_device(self, device):
        self.devices.append(device)

    def operate_devices(self):
        for device in self.devices:
            device.operate()

    def add_painting(self, painting):
        self.paintings.append(painting)

    def display_paintings(self):
        for painting in self.paintings:
            painting.display()

    def create_iterator(self):
        return DeviceIterator(self.devices)

    def set_state(self, walls, doors, windows):
        self.walls = walls
        self.doors = doors
        self.windows = windows

    def create_state_memento(self):
        return RoomState(self.walls, self.doors, self.windows)

    def restore_state(self, state):
        self.walls = state.walls
        self.doors = state.doors
        self.windows = state.windows

    def accept_visitor(self, visitor):
        for device in self.devices:
            device.accept(visitor)


# Builder Pattern
class RoomDirector:
    def __init__(self, builder):
        self._builder = builder

    def build_room(self):
        self._builder.create_new_room()
        self._builder.set_walls()
        self._builder.set_doors()
        self._builder.set_windows()
        self._builder.add_light()
        self._builder.add_thermostat()
        return self._builder.room


# Update RoomBuilder
class RoomBuilder(ABC):
    def __init__(self, factory: SmartDeviceFactory):
        self.factory = factory
        self.room = None

    def create_new_room(self, room_type: str):
        room_name = f"{room_type} {id(self)}"
        self.room = Room(room_name)

    @abstractmethod
    def set_walls(self):
        pass

    @abstractmethod
    def set_doors(self):
        pass

    @abstractmethod
    def set_windows(self):
        pass

    def add_light(self):
        light = self.factory.create_smart_light()
        self.room.add_device(light)

    def add_thermostat(self):
        thermostat = self.factory.create_smart_thermostat()
        self.room.add_device(thermostat)


# Update LivingRoomBuilder
class LivingRoomBuilder(RoomBuilder):
    def create_new_room(self):
        super().create_new_room("Living Room")

    def set_walls(self):
        logger.info("Building living room walls")

    def set_doors(self):
        logger.info("Building living room doors")

    def set_windows(self):
        logger.info("Building living room windows")


class KitchenRoomBuilder(RoomBuilder):
    def create_new_room(self):
        super().create_new_room("Kitchen")

    def set_walls(self):
        logger.info("Building kitchen room walls")

    def set_doors(self):
        logger.info("Building kitchen room doors")

    def set_windows(self):
        logger.info("Building kitchen room windows")


[chain_of_responsibility.py]

# ChainOfResponsibility.py
import logging

from SmartHomeSystem import Room, LivingRoomBuilder, RoomBuilder, SmartDeviceFactory

logger = logging.getLogger(__name__)


class RequestableRoom(Room):
    def __init__(self, name):
        super().__init__(name)
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def process_request(self, request):
        for handler in self.handlers:
            handler.handle_request(request)


class RequestableLivingRoomBuilder(RequestableRoom, LivingRoomBuilder, RoomBuilder):
    def __init__(self, factory: SmartDeviceFactory):
        RoomBuilder.__init__(self, factory)

    def create_new_room(self):
        room_name = f"Living Room {id(self)}"
        self.room = RequestableRoom(room_name)


class Request:
    def __init__(self, message):
        self.message = message


class Handler:
    def __init__(self, successor=None):
        self.successor = successor

    def handle_request(self, request):
        pass


class SecurityHandler(Handler):
    def handle_request(self, request):
        if request.message == "Security Alert":
            logger.info(f"Security alert received")
        elif self.successor is not None:
            logger.info(f"SecurityHandler pass the request: {request}")
            self.successor.handle_request(request)


class MaintenanceHandler(Handler):
    def handle_request(self, request):
        if request.message == "Maintenance Request":
            logger.info(f"Maintenance request received")
        elif self.successor is not None:
            logger.info(f"MaintenanceHandler pass the request: {request}")
            self.successor.handle_request(request)


class NotificationHandler(Handler):
    def handle_request(self, request):
        if request.message == "Notification":
            logger.info(f"Notification received")
        elif self.successor is not None:
            logger.info(f"NotificationHandler pass the request: {request}")
            self.successor.handle_request(request)


[command.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartLight

logger = logging.getLogger(__name__)


class Command(ABC):
    @abstractmethod
    def execute(self):
        pass


class TurnOnCommand(Command):
    def __init__(self, light: SmartLight):
        self.light = light

    def execute(self):
        self.light.turn_on()


class TurnOffCommand(Command):
    def __init__(self, light: SmartLight):
        self.light = light

    def execute(self):
        self.light.turn_off()


class RemoteControl:
    def __init__(self):
        self.command = None

    def set_command(self, command):
        self.command = command

    def press_button(self):
        if self.command:
            self.command.execute()


[composite.py]

# SmartDeviceComposite.py
import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartDevice

logger = logging.getLogger(__name__)


class SmartDeviceGroup(SmartDevice):
    def __init__(self):
        self.devices = []

    def add_device(self, device):
        self.devices.append(device)

    def remove_device(self, device):
        self.devices.remove(device)

    def operate(self):
        logger.info(f"Operating the SmartDeviceGroup {id(self)}.")
        for device in self.devices:
            device.operate()


[decorator.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartDevice

logger = logging.getLogger(__name__)


class SmartDeviceDecorator(SmartDevice):
    def __init__(self, device):
        self.device = device

    def operate(self):
        self.device.operate()


class MotionSensor(SmartDeviceDecorator):
    def operate(self):
        super().operate()
        self.detect_motion()

    def detect_motion(self):
        logger.info("Motion detected!")


class VoiceControl(SmartDeviceDecorator):
    def operate(self):
        super().operate()
        self.activate_voice_control()

    def activate_voice_control(self):
        logger.info("Voice control activated!")


[facade.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartLight, SmartThermostat

logger = logging.getLogger(__name__)


class SmartHomeFacade:
    def __init__(self):
        self.devices = []

    def add_device(self, device):
        self.devices.append(device)

    def turn_all_lights_on_and_off(self):
        logger.info(f"Facade {id(self)} is operating devices")
        for device in self.devices:
            if isinstance(device, SmartLight):
                device.operate()

    def adjust_temperature(self, temperature):
        logger.info(f"Facade {id(self)} is operating devices")
        for device in self.devices:
            if isinstance(device, SmartThermostat):
                if temperature > 25:
                    device.increase_temperature()
                else:
                    device.decrease_temperature()


[factory.py]

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertCreator(ABC):
    @abstractmethod
    def create_alert(self):
        pass


class Alert(ABC):
    @abstractmethod
    def send(self):
        pass


class FireAlert(Alert):
    def send(self):
        logger.info("Fire Alert: Smoke detected in the house!")


class IntruderAlert(Alert):
    def send(self):
        logger.info("Intruder Alert: Motion detected in the house!")


class FireAlertCreator(AlertCreator):
    def create_alert(self):
        return FireAlert()


class IntruderAlertCreator(AlertCreator):
    def create_alert(self):
        return IntruderAlert()


[flyweight.py]

import logging
from dataclasses import dataclass
import hashlib

import numpy as np


logger = logging.getLogger(__name__)


# Painting.py
@dataclass
class Painting:
    title: str
    artist: str
    style: str
    medium: str

    def __post_init__(self):
        s = f"Title: {self.title}, Artist: {self.artist}, Style: {self.style}, Medium: {self.medium}"
        np.random.seed(int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10**8))
        self.content = np.random.rand(100, 100)

    def display(self):
        logger.info(f"ID {id(self)}")
        logger.info(self.__repr__())
        logger.info(f"Painting Content (first 3 elements) {self.content[0,:3]}")


# PaintingFactory.py
class PaintingFactory:
    def __init__(self):
        self.paintings = {}

    def get_painting(self, title, artist, style, medium):
        key = f"{title}_{artist}_{style}_{medium}"
        if key not in self.paintings:
            self.paintings[key] = Painting(title, artist, style, medium)
        return self.paintings[key]


[interpreter.py]

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Expression:
    def interpret(self, context):
        pass


class LightOnExpression(Expression):
    def __init__(self, light):
        self.light = light

    def interpret(self, context):
        if context == "ON":
            self.light.turn_on()
        else:
            logger.info("Invalid context for LightOnExpression.")


class LightOffExpression(Expression):
    def __init__(self, light):
        self.light = light

    def interpret(self, context):
        if context == "OFF":
            self.light.turn_off()
        else:
            logger.info("Invalid context for LightOffExpression.")


class ThermostatSetExpression(Expression):
    def __init__(self, thermostat, temperature):
        self.thermostat = thermostat
        self.temperature = temperature

    def interpret(self, context):
        if context.isdigit():
            self.thermostat.set_temperature(int(context))
        else:
            logger.info("Invalid context for ThermostatSetExpression.")


[iterator.py]

import logging

logger = logging.getLogger(__name__)


class DeviceIterator:
    def __init__(self, devices):
        self.devices = devices
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.devices):
            device = self.devices[self.index]
            self.index += 1
            return device
        else:
            raise StopIteration


[momento.py]

class RoomState:
    def __init__(self, walls, doors, windows):
        self.walls = walls
        self.doors = doors
        self.windows = windows


class Caretaker:
    def __init__(self):
        self.mementos = {}

    def add_memento(self, room_name, memento):
        self.mementos[room_name] = memento

    def get_memento(self, room_name):
        return self.mementos.get(room_name)


[observer.py]

import logging
from abc import ABC, abstractmethod
from SmartHomeSystem import Room

from SmartHomeSystem import SmartThermostat

logger = logging.getLogger(__name__)


class Observer(ABC):
    @abstractmethod
    def update(self, sender, event):
        pass


class AlertSystem(Observer):
    def update(self, sender: Room, event: str):
        logger.info(f"Alert! Room {id(sender)} reports a {event}!")


[prototype.py]

import copy
import logging

logger = logging.getLogger(__name__)


class SmartDevicePrototype:
    def clone(self):
        return copy.deepcopy(self)


[proxy.py]

import logging

from SmartHomeSystem import SmartDevice

logger = logging.getLogger(__name__)


class SmartDeviceProxy(SmartDevice):
    def __init__(self, device):
        self.device = device

    def operate(self):
        self._check_access()
        self.device.operate()

    def _check_access(self):
        logger.info("Access control check passed.")


[singleton.py]

import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import RoomBuilder, RoomDirector
from SmartHomeSystem import AlertSystem

logger = logging.getLogger(__name__)


# Meta class/Monostate method
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Singleton SmartHome
class SmartHome(metaclass=Singleton):
    def __init__(self):
        if not hasattr(self, "initiated"):
            self.rooms = {}
            self.alert_system = AlertSystem()
            self.initiated = True
            self.home_lock = None
            logger.info("Initialized the smart home.")

    def add_room(self, builder: RoomBuilder):
        director = RoomDirector(builder)
        room = director.build_room()
        self.rooms[room.name] = room

    def add_alert_system_to_room_devices(self, room_name: str):
        room = self.rooms.get(room_name)
        if room:
            for device in room.devices:
                device.add_observer(self.alert_system)
        else:
            logger.warning(f"No room found with name {room_name}")

    def operate_room(self, room_name: str):
        room = self.rooms.get(room_name)
        if room:
            room.operate_devices()
        else:
            logger.warning(f"No room found with name {room_name}")

    def __str__(self):
        room_names = ", ".join(self.rooms.keys())
        return f"SmartHome with rooms: {room_names}"


""" using __new__
class SmartHome:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SmartHome, cls).__new__(cls)
            cls._instance.__initiated = False
        return cls._instance

    def init(self):
        if self.__initiated:
            return
        self.rooms = []
        self.devices = []
        self.__initiated = True
"""

if __name__ == "__main__":
    home1 = SmartHome()
    home2 = SmartHome()
    assert home1 is home2


[state.py]

import logging

from SmartHomeSystem import SmartDevice

logger = logging.getLogger(__name__)


class SmartLockState:
    def enter_pin(self, pin, lock):
        pass

    def lock(self, lock):
        pass

    def unlock(self, lock):
        pass


class LockedState(SmartLockState):
    def enter_pin(self, pin, lock):
        if pin == lock.correct_pin:
            logger.info("Correct PIN entered. Unlocking the smart lock.")
            lock.change_state(UnlockedState())
        else:
            logger.info("Incorrect PIN entered. The smart lock remains locked.")

    def lock(self, lock):
        logger.info("The smart lock is already locked.")

    def unlock(self, lock):
        logger.info("Unlock the smart lock first before trying to lock it again.")


class UnlockedState(SmartLockState):
    def enter_pin(self, pin, lock):
        logger.info("The smart lock is already unlocked.")

    def lock(self, lock):
        logger.info("Locking the smart lock.")
        lock.change_state(LockedState())

    def unlock(self, lock):
        logger.info("The smart lock is already unlocked.")


class SmartLock(SmartDevice):
    def __init__(self, correct_pin):
        self.correct_pin = correct_pin
        self.state = LockedState()

    def change_state(self, state):
        self.state = state

    def enter_pin(self, pin):
        self.state.enter_pin(pin, self)

    def lock(self):
        self.state.lock(self)

    def unlock(self):
        self.state.unlock(self)


[strategy.py]

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TemperatureControlStrategy(ABC):
    def __init__(self, increase_func, decrease_func) -> None:
        self.increase_func = increase_func
        self.decrease_func = decrease_func

    @abstractmethod
    def control_temperature(self):
        pass


class NormalTemperatureControl(TemperatureControlStrategy):
    def control_temperature(self):
        logger.info("Applying normal temperature control.")


class EcoTemperatureControl(TemperatureControlStrategy):
    def control_temperature(self):
        logger.info("Applying eco temperature control.")


[visitor.py]

# VisitorPattern.py
import logging
from abc import ABC, abstractmethod

from SmartHomeSystem import SmartDevice, SmartLight, SmartThermostat

logger = logging.getLogger(__name__)


class DeviceVisitor(ABC):
    def visit(self, device: SmartDevice):
        logger.info(f"Visitor {id(self)} visiting device {id(device)}")
        if isinstance(device, SmartThermostat):
            self.visit_smart_thermostat(device)
        if isinstance(device, SmartLight):
            self.visit_smart_light(device)

    @abstractmethod
    def visit_smart_light(self, smart_light):
        pass

    @abstractmethod
    def visit_smart_thermostat(self, smart_thermostat):
        pass


class DeviceStatusVisitor(DeviceVisitor):
    def visit_smart_light(self, smart_light):
        logger.info("Checking status of Smart Light.")
        smart_light.operate()

    def visit_smart_thermostat(self, smart_thermostat):
        logger.info("Checking status of Smart Thermostat.")
        smart_thermostat.operate()
