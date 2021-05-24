from slashingbeaconclient import BlockSlashingBeaconClient
from beaconclient import BeaconClient
from multiprocessing import JoinableQueue, Queue
from typing import List, Optional
from validator import ValidatorBuilder
from builder import Builder


class BeaconClientBuilder(Builder):
    configpath: str
    configname: str

    debug: bool
    debugfile: Optional[str]
    profile: bool

    validators_count: int
    validator_builders: List[ValidatorBuilder]

    neccessary_info_set: bool
    validator_start_at: int
    recv_queue: JoinableQueue
    send_queue: Queue
    mode: str

    def __init__(self, configpath, configname, parent_builder=None,):
        super(BeaconClientBuilder, self).__init__(parent_builder)
        self.validator_builders = []
        self.configpath = configpath
        self.configname = configname
        self.debug = False
        self.debugfile = None
        self.profile = False
        self.mode = 'HONEST'
        self.neccessary_info_set = False
        self.validator_start_at = 0
        self.simulator_to_client_queue = JoinableQueue()
        self.client_to_simulator_queue = Queue()

    def build_impl(self, counter):
        if not self.neccessary_info_set:
            raise ValueError('Need to specify queues and validator start index')
        if self.mode not in ('HONEST', 'BlockSlashing'):
            raise ValueError(f'Unknown mode: {self.mode}')

        if self.mode == 'HONEST':
            return BeaconClient(
                counter=counter,
                simulator_to_client_queue=self.simulator_to_client_queue,
                client_to_simulator_queue=self.client_to_simulator_queue,
                configpath=self.configpath,
                configname=self.configname,
                validator_builders=self.validator_builders,
                validator_first_counter=self.validator_start_at,
                debug=self.debug,
                profile=self.profile
            )
        elif self.mode == 'BlockSlashing':
            print('CONSTRUCT BLOCK SLASHER')
            return BlockSlashingBeaconClient(
                counter=counter,
                simulator_to_client_queue=self.simulator_to_client_queue,
                client_to_simulator_queue=self.client_to_simulator_queue,
                configpath=self.configpath,
                configname=self.configname,
                validator_builders=self.validator_builders,
                validator_first_counter=self.validator_start_at,
                debug=self.debug,
                profile=self.profile
            )

    def register(self, child_builder: ValidatorBuilder):
        child_builder.validators_count = int(self.validators_count)
        self.validator_builders.append(child_builder)

    def set_debug(self, flag=False):
        self.debug = flag
        return self

    def set_profile(self, flag=False):
        self.profile = flag
        return self

    def set_mode(self, mode):
        self.mode = mode
        return self

    def validators(self, count):
        self.validators_count = count
        return ValidatorBuilder(parent_builder=self)

    def neccessary_information(self, validator_start_at, client_to_simulator_queue):
        self.neccessary_info_set = True
        self.validator_start_at = validator_start_at
        self.client_to_simulator_queue = client_to_simulator_queue